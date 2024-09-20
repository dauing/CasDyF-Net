# CasDyF-Net: Image Dehazing via Cascaded Dynamic Filters
  
Yinglong Wang, Bin He

>Image dehazing aims to restore image clarity and visual quality by reducing atmospheric scattering and absorption effects. While deep learning has made significant strides in this area, more and more methods are constrained by network depth. Consequently, lots of approaches have adopted parallel branching strategies. however, they often prioritize aspects such as resolution, receptive field, or frequency domain segmentation without dynamically partitioning branches based on the distribution of input features. Inspired by dynamic filtering, we propose using cascaded dynamic filters to create a multi-branch network by dynamically generating filter kernels based on feature map distribution. To better handle branch features, we propose a residual multiscale block (RMB), combining different receptive fields. Furthermore, We also introduce a dynamic convolution-based local fusion method to merge features from adjacent branches. Experiments on RESIDE, Haze4K, and O-Haze datasets validate our method’s effectiveness, with our model achieving a PSNR of 43.21dB on the RESIDE-Indoor dataset.

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~
## Training and Evaluation

## Results [Download](https://drive.google.com/)
|Dataset|PSNR|SSIM|
|------|-----|----|
|SOTS-Indoor|43.21|0.997|
|SOTS-Outdoor|38.94|0.997|
|Dense-Haze|17.13|0.65|
|O-HAZE|20.55|0.81|
|Haze4K|34.12|0.99|

## Citation

## Contact
Yinglong Wang, e-mail:wangyinglong2023@gmail.com, wechat:dauing2023