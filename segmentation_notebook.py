import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import os
    import matplotlib.pyplot as plt
    import SimpleITK as sitk
    import numpy as np
    import pandas as pd
    from glob import glob
    import nibabel as nib
    import torch
    import torch.nn as nn
    import monai
    from monai.data import PILReader, ImageDataset, create_test_image_3d, decollate_batch, DataLoader, Dataset
    from monai.inferers import sliding_window_inference
    from monai.config import KeysCollection
    from monai.data.image_reader import ITKReader
    from monai.metrics import DiceMetric
    from monai.transforms import Activations, EnsureChannelFirst, AsDiscrete, Compose, RandSpatialCrop, ScaleIntensity, LoadImage, Resize, SaveImage, ToTensor, EnsureChannelFirstd, ScaleIntensityd, LoadImaged, Resized, ToTensord
    from monai.networks.nets import UNETR
    from monai.visualize import plot_2d_or_3d_image
    from tqdm import tqdm
    import random
    from sklearn.model_selection import train_test_split
    from typing import Dict, Hashable, Mapping  #For dictionary workflow
    return (
        Activations,
        AsDiscrete,
        Compose,
        DataLoader,
        Dataset,
        DiceMetric,
        EnsureChannelFirstd,
        ITKReader,
        LoadImage,
        Resized,
        ScaleIntensityd,
        ToTensord,
        UNETR,
        nn,
        np,
        os,
        plt,
        sitk,
        torch,
        tqdm,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experimental Visualizations of Data
    """)
    return


@app.cell
def _():
    data_dir = 'data/PROMISE12/training_data/'

    test_mhd_img_path = 'Case00.mhd'


    test_mhd_path = data_dir + test_mhd_img_path
    return data_dir, test_mhd_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using Monai's image loader
    """)
    return


@app.cell
def _(LoadImage, test_mhd_path):
    load_img = LoadImage(image_only=True)#, reader=ITKReader())
    monai_img = load_img(test_mhd_path)
    return (monai_img,)


@app.cell
def _(monai_img):
    print(monai_img.shape)
    return


@app.cell
def _(monai_img, plt):
    i=25
    plt.imshow(monai_img[:, :, i], cmap="gray")
    plt.title(f"Slice {i} using Monai LoadImage")
    return (i,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using SimpleITK
    """)
    return


@app.cell
def _(sitk, test_mhd_path):
    img = sitk.ReadImage(test_mhd_path)

    img_array = sitk.GetArrayFromImage(img)
    return (img_array,)


@app.cell
def _(i, img_array, plt):
    plt.imshow(img_array[i], cmap="gray")
    plt.title(f"Slice {i} using SimpleITK")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Visualizing a Mask
    """)
    return


@app.cell
def _(data_dir):
    test_mhd_mask_path = 'Case00_segmentation.mhd'

    test_mask_path = data_dir + test_mhd_mask_path
    return (test_mask_path,)


@app.cell
def _(ITKReader, LoadImage, test_mask_path):
    load_mask = LoadImage(image_only=True, reader=ITKReader())
    mask = load_mask(test_mask_path)
    return


@app.cell
def _(sitk, test_mask_path):
    mask_img = sitk.ReadImage(test_mask_path)

    mask_array = sitk.GetArrayFromImage(mask_img)
    return (mask_array,)


@app.cell
def _(mask_array):
    mask_array.shape
    return


@app.cell
def _(i, mask_array, plt):
    plt.imshow(mask_array[i], cmap="gray")
    plt.title(f"Slice {i} Mask")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Loading to prepare for training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TO do:

    Reminder to convert the shape from MONAI ITK from H,W,D to D,H,W to match the mask format.
    OR make use of SImpleITK for reading in the images, and then pushing to MONAI for data processing and training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    MONAI will require the arrays to be in tensors so use .ToTensor() transformation

    It is part of Monai

    from monai.transforms import ToTensor

    [Monai Documenation on Transforms](https://docs.monai.io/en/stable/transforms.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### UNETR
    """)
    return


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    return (device,)


@app.cell
def _(np, sitk):
    class Load_MHD:
        def __init__(self, keys):
            self.keys = keys

        def __call__(self, data):
            for key in self.keys:

                if isinstance(data[key], str):
                    img = sitk.ReadImage(data[key])
                    if key == "label":
                        arr = sitk.GetArrayFromImage(img).astype(np.float32)
                        data[key] = (arr >= 0.5).astype(np.uint8)
                    else:
                        data[key] = sitk.GetArrayFromImage(img).astype(np.float32)
            return data
    return (Load_MHD,)


@app.cell
def _(
    Compose,
    EnsureChannelFirstd,
    Load_MHD,
    Resized,
    ScaleIntensityd,
    ToTensord,
):
    image_size = (256, 256, 64)

    train_transforms = Compose([

        # LoadImaged(keys=["image", "label"], reader=SimpleITKReader()),
        Load_MHD(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=image_size, mode=["area", "nearest"],),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        # LoadImaged(keys=["image", "label"]),
        Load_MHD(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=image_size, mode=["area", "nearest"],),
        ToTensord(keys=["image", "label"]),
    ])
    return train_transforms, val_transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Model
    """)
    return


@app.cell
def _(UNETR, device):
    model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(256, 256, 64), 
        feature_size=16,
        hidden_size=256,
        mlp_dim=512,
        num_heads=4,
        norm_name='instance',
        dropout_rate=0.1
    ).to(device)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Data loading
    """)
    return


@app.cell
def _(data_dir, os):
    case_ids = sorted({f[:6] for f in os.listdir(data_dir) if f.endswith(".mhd") and "_segmentation" not in f})

    #this should loop through the data to grab the names and sort them alphanumerically CaseXX
    return (case_ids,)


@app.cell
def _(case_ids, data_dir, os):
    all_files = [
        {
            "image": os.path.join(data_dir, f"{case_id}.mhd"),
            "label": os.path.join(data_dir, f"{case_id}_segmentation.mhd")
        }
        for case_id in case_ids
    ]
    return (all_files,)


@app.cell
def _(
    DataLoader,
    Dataset,
    all_files,
    train_test_split,
    train_transforms,
    val_transforms,
):
    batch_size = 1

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_files, train_loader, val_files, val_loader


@app.cell
def _(train_files):
    print(train_files)
    return


@app.cell
def _(val_files):
    print(val_files)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Training Preparation
    """)
    return


@app.cell
def _(Activations, AsDiscrete, Compose, DiceMetric):
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    post_pred = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.3),
    ])

    post_label = AsDiscrete(threshold=0.3)
    return dice_metric, post_label, post_pred


@app.cell
def _(model, nn, torch):
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    return epochs, loss_function, optimizer


@app.cell
def _():
    train_losses = []
    val_losses = []
    return train_losses, val_losses


@app.cell
def _(
    device,
    dice_metric,
    epochs,
    loss_function,
    model,
    optimizer,
    post_label,
    post_pred,
    torch,
    tqdm,
    train_loader,
    train_losses,
    val_loader,
    val_losses,
):
    for epoch in range(epochs):
        model.train()
        _epoch_train_loss = 0
        for _batch in tqdm(train_loader, desc=f'Epoch {epoch + 1} - Training'):
            _images = _batch['image'].float().to(device)
            _labels = _batch['label'].float().to(device)
            _outputs = model(_images)
            _loss = loss_function(_outputs, _labels)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()
            _epoch_train_loss = _epoch_train_loss + _loss.item()
        _avg_train_loss = _epoch_train_loss / len(train_loader)
        train_losses.append(_avg_train_loss)
        print(f'Training Loss: {_avg_train_loss:.4f}')
        model.eval()
        _epoch_val_loss = 0
        dice_metric.reset()
        with torch.no_grad():
            for _batch in tqdm(val_loader, desc=f'Epoch {epoch + 1} - Validation'):
                _val_images = _batch['image'].float().to(device)
                _val_labels = _batch['label'].float().to(device)
                _val_outputs = model(_val_images)
                _val_loss = loss_function(_val_outputs, _val_labels)
                _epoch_val_loss = _epoch_val_loss + _val_loss.item()
                _val_outputs_post = post_pred(_val_outputs)
                _val_labels_post = post_label(_val_labels)
                dice_metric(_val_outputs_post, _val_labels_post)
        _avg_val_loss = _epoch_val_loss / len(val_loader)
        _val_dice, _ = dice_metric.aggregate()
        dice_metric.reset()
        val_losses.append(_avg_val_loss)
        print(f'Validation Loss: {_avg_val_loss:.4f} - Dice: {_val_dice.item():.4f}')
    return (epoch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To do: model this more to match literature instead of documentation
    """)
    return


@app.cell
def _(
    device,
    dice_metric,
    epoch,
    loss_function,
    model,
    plt,
    post_label,
    post_pred,
    torch,
    tqdm,
    val_loader,
):
    with torch.no_grad():
        for _batch in tqdm(val_loader, desc=f'Epoch {epoch + 1} - Validation'):
            _val_images = _batch['image'].float().to(device)
            _val_labels = _batch['label'].float().to(device)
            _val_outputs = model(_val_images)
            _val_loss = loss_function(_val_outputs, _val_labels)
            _epoch_val_loss = _epoch_val_loss + _val_loss.item()
            _val_outputs_post = post_pred(_val_outputs)
            _val_labels_post = post_label(_val_labels)
            dice_metric(_val_outputs_post, _val_labels_post)
            z = _val_outputs_post.shape[4] // 2
            pred = _val_outputs_post[0, 0, :, :, z].cpu().numpy()
            label = _val_labels_post[0, 0, :, :, z].cpu().numpy()
            plt.subplot(1, 2, 1)
            plt.imshow(pred > 0.3, cmap='gray')
            plt.title('Predicted')
            plt.subplot(1, 2, 2)
            plt.imshow(label, cmap='gray')
            plt.title('Ground Truth')
            plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DV-VNet
    """)
    return


@app.cell
def _(torch):
    device_1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device_1)
    return (device_1,)


@app.cell
def _():
    from monai.networks.nets import ViT
    from monai.networks.blocks import UnetOutBlock
    from monai.networks.layers import Norm
    from monai.utils import set_determinism
    return UnetOutBlock, ViT


@app.cell
def _(UnetOutBlock, ViT, nn):
    class DVVNet(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, img_size=(256, 256, 64), patch_size=(16, 16, 16)):
            super().__init__()
            self.img_size=img_size
            self.patch_size=patch_size
            self.vit = ViT(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=256,
                mlp_dim=256,
                num_heads=4,
                dropout_rate=0.1,
                spatial_dims=3
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=2, stride=(2, 2, 2)),  
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True),

                # on the updscaling we are going to go from 32 to 64 in all dimensions

                nn.ConvTranspose3d(128, 64, kernel_size=2, stride=(2, 2, 2)),   
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),

                #the next two will only work on HxW of image since I have set it to be 256x256x64 slices

                nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),

                nn.ConvTranspose3d(32, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
            )



            self.out = UnetOutBlock(spatial_dims=3, in_channels=16, out_channels=out_channels)
            # self.out = UnetOutBlock(spatial_dims=3, in_channels=32, out_channels=out_channels)

        def forward(self, x):
            x = self.vit(x)
            if isinstance(x, tuple):
                x = x[0] 

            B, N, C = x.shape
            D, H, W = [i // p for i, p in zip(self.img_size, self.patch_size)]
            x = x.view(B, C, D, H, W)

            x = self.decoder(x)
            return self.out(x)
    return (DVVNet,)


@app.cell
def _(torch):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return


@app.cell
def _(DVVNet, device_1, nn, torch):
    model_1 = DVVNet().to(device_1)
    loss_function_1 = nn.BCEWithLogitsLoss()
    optimizer_1 = torch.optim.Adam(model_1.parameters(), 0.001)
    epochs_1 = 2
    return epochs_1, loss_function_1, model_1, optimizer_1


@app.cell
def _(Activations, AsDiscrete, Compose, DiceMetric):
    dice_metric_1 = DiceMetric(include_background=True, reduction='mean', get_not_nans=True)
    post_pred_1 = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.3)])
    post_label_1 = AsDiscrete(threshold=0.3)
    return dice_metric_1, post_label_1, post_pred_1


@app.cell
def _():
    train_losses_1 = []
    val_losses_1 = []
    return train_losses_1, val_losses_1


@app.cell
def _(
    device_1,
    dice_metric_1,
    epochs_1,
    loss_function_1,
    model_1,
    optimizer_1,
    post_label_1,
    post_pred_1,
    torch,
    tqdm,
    train_loader,
    train_losses_1,
    val_loader,
    val_losses_1,
):
    for epoch_1 in range(epochs_1):
        model_1.train()
        _epoch_train_loss = 0
        for _batch in tqdm(train_loader, desc=f'Epoch {epoch_1 + 1} - Training'):
            _images = _batch['image'].float().to(device_1)
            _labels = _batch['label'].float().to(device_1)
            _outputs = model_1(_images)
            _loss = loss_function_1(_outputs, _labels)
            optimizer_1.zero_grad()
            _loss.backward()
            optimizer_1.step()
            _epoch_train_loss = _epoch_train_loss + _loss.item()
        _avg_train_loss = _epoch_train_loss / len(train_loader)
        train_losses_1.append(_avg_train_loss)
        print(f'Training Loss: {_avg_train_loss:.4f}')
        model_1.eval()
        _epoch_val_loss = 0
        dice_metric_1.reset()
        with torch.no_grad():
            for _batch in tqdm(val_loader, desc=f'Epoch {epoch_1 + 1} - Validation'):
                _val_images = _batch['image'].float().to(device_1)
                _val_labels = _batch['label'].float().to(device_1)
                _val_outputs = model_1(_val_images)
                _val_loss = loss_function_1(_val_outputs, _val_labels)
                _epoch_val_loss = _epoch_val_loss + _val_loss.item()
                _val_outputs_post = post_pred_1(_val_outputs)
                _val_labels_post = post_label_1(_val_labels)
                dice_metric_1(_val_outputs_post, _val_labels_post)
        _avg_val_loss = _epoch_val_loss / len(val_loader)
        _val_dice, _ = dice_metric_1.aggregate()
        dice_metric_1.reset()
        val_losses_1.append(_avg_val_loss)
        print(f'Validation Loss: {_avg_val_loss:.4f} - Dice: {_val_dice.item():.4f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Path sizes of 16 and 8 seem to work
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When using patch of 4,4,4

    OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 GiB. GPU 0 has a total capacity of 24.00 GiB of which 5.83 GiB is free. Of the allocated memory 16.85 GiB is allocated by PyTorch, and 32.58 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

