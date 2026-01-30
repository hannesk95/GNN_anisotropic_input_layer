import torch
from MedConv3D import MedConv3D
torch.nn.Conv3d = MedConv3D

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
RandAffined,
RandFlipd,
RandScaleIntensityd,
RandShiftIntensityd
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
from torch.optim.lr_scheduler import PolynomialLR, ExponentialLR, OneCycleLR, CyclicLR
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
import mlflow

directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
data_dir = os.path.join(root_dir, "Task09_Spleen")
data_dir = "/home/johannes/Code/GNN_anisotropic_input_layer/Task09_Spleen"  # --- IGNORE ---
if not os.path.exists(data_dir):
    download_and_extract(resource, compressed_file, data_dir, md5)

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-9], data_dicts[-9:]

set_determinism(seed=0)

patch_size = (160, 160, 32)
# 'during', 'after', 'normal'

mlflow.set_experiment("MedConv3D_spleen_segmentation_monai")

for spacings_type in [ 'normal_resample']:

    mlflow.end_run() 
    with mlflow.start_run(run_name=f"{spacings_type}"):

        os.environ["MEDCONV3D_SPACINGS_TYPE"] = spacings_type
        if spacings_type == 'normal_resample':
            os.environ["MEDCONV3D_SPACINGS_TYPE"] = "normal"

        mlflow.log_param("spacings_type", spacings_type)
        resample_transform = Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 5.0), mode=("bilinear", "nearest"))
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                resample_transform if spacings_type == 'normal_resample' else lambda x: x,
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=patch_size,
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                # user can also add other random transforms
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=0.0, spatial_size=patch_size,
                    rotate_range=(0, 0, np.pi/15),
                    scale_range=(0.1, 0.1, 0.1)),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-57,
                    a_max=164,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                resample_transform if spacings_type == 'normal_resample' else lambda x: x,
            ]
        )




        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0, num_workers=4)
        # train_ds = Dataset(data=train_files, transform=train_transforms)

        # use batch_size=2 to load images and use RandCropByPosNegLabeld
        # to generate 2 x 4 images for network training
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0, num_workers=4)
        # val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)




        # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
        device = torch.device("cuda:0")
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.INSTANCE,
        ).to(device)
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        lr_scheduler = PolynomialLR(optimizer, total_iters=1000, power=0.9)

        max_epochs = 1000
        iterations_per_epoch = 25
        val_interval = 2
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])

        iter_loader = iter(train_loader)

        # spacings = []
        # for a in iter_loader:
        #     spacing = tuple(a["image"].pixdim[0])
        #     print(f"Example image spacing: {spacing}")
        #     spacings.append(spacing)
        
        # spacings = np.array(spacings)
        # median_spacings = np.median(spacings, axis=0)
        

        # print("median spacing : " + str(median_spacings))
        # exit()

        spacings_list = []
        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for i in range(iterations_per_epoch):
                try:
                    batch_data = next(iter_loader)
                except Exception as e:
                    print(e)
                    iter_loader = iter(train_loader)
                    batch_data = next(iter_loader)

                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                spacing = str(batch_data["image"].pixdim[0].tolist())[1:-1].replace(" ","")
                os.environ["MEDCONV3D_SPACINGS"] = spacing

                if epoch == 0:
                    spacings_list.append(spacing)
                
                if epoch == 1 and step == 0:            
                    mlflow.log_param("spacings_used", spacings_list)


                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            mlflow.log_metric("train_loss", epoch_loss, step=epoch + 1)

            print("LR scheduler lr : " + str(lr_scheduler.get_last_lr()))
            lr_scheduler.step()

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        sw_batch_size = 4
                        val_outputs = sliding_window_inference(val_inputs, patch_size, sw_batch_size, model)
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)

                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    metric_values.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )
            
                mlflow.log_metric("val_mean_dice", metric, step=epoch + 1)
                

        print(f"train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")


        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel("epoch")
        plt.plot(x, y)
        plt.show()

