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
- [ ] Model Implementation:
  - [ ] U-Net
  - [ ] V-Net
  - [ ] U-NetR
  - [ ] DT-VNet
- [ ] Model training 
- [ ] Interpretability integration (Grad-CAM)



## References

1. Litjens G, Toth R, van de Ven W, Hoeks C, Kerkstra S, van Ginneken B, et al. Evaluation of prostate segmentation algorithms for MRI: The PROMISE12 challenge. Medical Image Analysis. 2014 Feb 1;18(2):359–73. 

2. Singla D, Cimen F, Narasimhulu CA. Novel artificial intelligent transformer U-NET for better identification and management of prostate cancer. Mol Cell Biochem. 2023 Jul 1;478(7):1439–45. 

3. Bhandary S, Kuhn D, Babaiee Z, Fechter T, Benndorf M, Zamboglou C, et al. Investigation and benchmarking of U-Nets on prostate segmentation tasks. Computerized Medical Imaging and Graphics. 2023 Jul 1;107:102241. 

4. Cai Y, Lu H, Wu S, Berretti S, Wan S. DT-VNet: Deep Transformer-based VNet Framework for 3D Prostate MRI Segmentation. IEEE Journal of Biomedical and Health Informatics. 2024;1–8. 

5. Zhihao Sun, Jianhong Liu, Xiaoming Chen, Guangyu Wang, Lihua Zhang, Meilin Zhou. Leveraging Pretrained Transformers for Medical Text Summarization. 2025 [cited 2025 May 15]; Available from: https://rgdoi.net/10.13140/RG.2.2.32014.06721

6. Huang G, Xia B, Zhuang H, Yan B, Wei C, Qi S, et al. A Comparative Analysis of U-Net and Vision Transformer Architectures in Semi-Supervised Prostate Zonal Segmentation. Bioengineering. 2024 Sep;11(9):865. 

7. Yan Y, Liu R, Chen H, Zhang L, Zhang Q. CCT-Unet: A U-Shaped Network Based on Convolution Coupled Transformer for Segmentation of Peripheral and Transition Zones in Prostate MRI. IEEE Journal of Biomedical and Health Informatics. 2023 Sep;27(9):4341–51. 





