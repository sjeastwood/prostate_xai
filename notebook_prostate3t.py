import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a notebook for a later project that will do zonal detection/segmentation on the prostate
    """)
    return


@app.cell
def _():
    import monai
    import numpy as np
    import pandas as pd
    import pydicom

    from monai.data import PILReader
    from monai.transforms import LoadImage, LoadImaged, Resized, Compose, SaveImage
    from monai.config import print_config
    return LoadImage, monai


@app.cell
def _():
    files = '/mnt/i/prostate_data/prostate3t/manifest-La6DeDkm568029047075601590/Prostate-3T/Prostate3T-01-0001/03-15-2003-NA-MR prostaat kanker detectiemc MCAPRODET-79232/2.000000-t2tsetra-49698/1-01.dcm'
    return (files,)


@app.cell
def _(LoadImage, files):
    data = LoadImage(image_only=True)(files)
    return (data,)


@app.cell
def _(data):
    print(data.shape)
    return


@app.cell
def _(data, monai):
    fig = monai.visualize.matshow3d(data)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
