# prostate_xai


## Project Description and Goal
Taking first the [PROMISE12](https://doi.org/10.1016/j.media.2013.12.002) dataset containing MRI slices of prostate with segmentation masks. The goal is to implement models from literature, attempt training for better performance, and then implement explainable AI to analyze what is important for the model.


## Project Setup

1. Set up a python3.11 environment with venv or miniconda

2. Activate environment and 

        pip install -r requirements.txt

3. Download the dataset files

        bash download_promise12.sh

4. Execute code in segmentation_notebook.ipynb



## Goals
- [x] Dataset loaded and visualized
- [ ] Literature review 
- [ ] Model Implementation
- [ ] Model training 
- [ ] Interpretability integration (Grad-CAM)



## References

1. Litjens G, Toth R, van de Ven W, Hoeks C, Kerkstra S, van Ginneken B, et al. Evaluation of prostate segmentation algorithms for MRI: The PROMISE12 challenge. Medical Image Analysis. 2014 Feb 1;18(2):359–73. 

2. Huang G, Xia B, Zhuang H, Yan B, Wei C, Qi S, et al. A Comparative Analysis of U-Net and Vision Transformer Architectures in Semi-Supervised Prostate Zonal Segmentation. Bioengineering. 2024 Sep;11(9):865. 

3. Yan Y, Liu R, Chen H, Zhang L, Zhang Q. CCT-Unet: A U-Shaped Network Based on Convolution Coupled Transformer for Segmentation of Peripheral and Transition Zones in Prostate MRI. IEEE Journal of Biomedical and Health Informatics. 2023 Sep;27(9):4341–51. 
