# CasDyF-Net: Image Dehazing via Cascaded Dynamic Filters
  
**Authors**: Yinglong Wang, Bin He

**Paper link**: [CasDyF-Net: Image Dehazing via Cascaded Dynamic Filters](https://arxiv.org/abs/2409.08510)

 **Abstract:**

>Image dehazing aims to restore image clarity and visual quality by reducing atmospheric scattering and absorption effects. While deep learning has made significant strides in this area, more and more methods are constrained by network depth. Consequently, lots of approaches have adopted parallel branching strategies. however, they often prioritize aspects such as resolution, receptive field, or frequency domain segmentation without dynamically partitioning branches based on the distribution of input features. Inspired by dynamic filtering, we propose using cascaded dynamic filters to create a multi-branch network by dynamically generating filter kernels based on feature map distribution. To better handle branch features, we propose a residual multiscale block (RMB), combining different receptive fields. Furthermore, We also introduce a dynamic convolution-based local fusion method to merge features from adjacent branches. Experiments on RESIDE, Haze4K, and O-Haze datasets validate our method’s effectiveness, with our model achieving a PSNR of 43.21dB on the RESIDE-Indoor dataset.

## Installation
The project is built with **PyTorch 3.8**, **PyTorch 1.10.0**, **CUDA 11.3**,

For installing, follow these instructions:
~~~
pip install -r requirements.txt
~~~
## Training and Evaluation
This part will be provided in the final version.
## Results 
You can download the pre-trained **models** from this link:

[Download the pre-trained models](https://drive.google.com/drive/folders/10zPlf5OPEz-VCO7HiAnbCNGgp29K2hOC?usp=drive_link)

If you just need the output images of the models, you can download the **images** from this link:

[Download the images](https://drive.google.com/drive/folders/1jbgU3fwJ3gZfprJcsyuCToPcD6eReDK4?usp=drive_link)

|Dataset|PSNR|SSIM|
|------|-----|----|
|SOTS-Indoor|43.21|0.997|
|SOTS-Outdoor|38.86|0.995|
|Dense-Haze|17.13|0.65|
|O-HAZE|20.55|0.81|
|Haze4K|35.73|0.99|

## Citation
Will be provided when the paper is accepted.
## Contact
name: Yinglong Wang

e-mail: wangyinglong2023@gmail.com

wechat: dauing2023