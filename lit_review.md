# Literature Review - In Progress

There is a struggle with differences in MRI images in terms of scanner manufacturer, magnetic field strength and image acquisition protocol [1]. The challenge was set up to see if someone can make use of data acquired from 4 different centers. This challenge resulted in a research lab acquiring 85.72. They had an efficient run time of 8 minutes and 3 seconds per case. 

The standard metric used for segmentation task is the Dice coefficient for the slices, and absolute relative volume difference for the 3dimensional images.

The winning entries made use of image details to identify landmarks that were important. This can allow someone to spring off this concept and move towards machine learning.

Basic U-net has been applied to PROMISE12 in [2]. They found Attention U-Net was suprior. This leads to other models such as V-Net which makes uses of residual blocks while downsampling. The introduction of the Vision Transformer then helps pave the way for these models to combine [3][4][5]. 


## References

1. Litjens G, Toth R, van de Ven W, Hoeks C, Kerkstra S, van Ginneken B, et al. Evaluation of prostate segmentation algorithms for MRI: The PROMISE12 challenge. Medical Image Analysis. 2014 Feb 1;18(2):359–73. 

2. Bhandary S, Kuhn D, Babaiee Z, Fechter T, Benndorf M, Zamboglou C, et al. Investigation and benchmarking of U-Nets on prostate segmentation tasks. Computerized Medical Imaging and Graphics. 2023 Jul 1;107:102241. 

3. Singla D, Cimen F, Narasimhulu CA. Novel artificial intelligent transformer U-NET for better identification and management of prostate cancer. Mol Cell Biochem. 2023 Jul 1;478(7):1439–45. 

4. Hatamizadeh A, Tang Y, Nath V, Yang D, Myronenko A, Landman B, et al. UNETR: Transformers for 3D Medical Image Segmentation. In 2022 [cited 2025 May 15]. p. 574–84. Available from: https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html

5. Cai Y, Lu H, Wu S, Berretti S, Wan S. DT-VNet: Deep Transformer-based VNet Framework for 3D Prostate MRI Segmentation. IEEE Journal of Biomedical and Health Informatics. 2024;1–8. 
