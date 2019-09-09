# ADCF_2019
code for Adaptive Target Tracking Algorithm with Dynamic Weighted Hierarchical Convolutional Neural Network Feature.

## 1. Intro
 - This project includes source code for first experience in our paper based on PyTorch and usage of CNN features.
 - For models, inlcuding mainstream models from torchvision [(PyTorch)](https://pytorch.org/docs/stable/torchvision/models.html) in 
 CNNFeat. Models from tensorflow or caffe should perform better on visual tracking, such as [(pretrained-models)](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) or [(matconvnet)](https://github.com/vlfeat/matconvnet).
## 2. Install
numpy==1.14.5 
torch==0.4.0 
torchvision==0.4.0 
scipy==1.0.0 
opencv_python==3.4.2 
matplotlib==2.1.2 
numba==0.43.1 
pillow==6.1.0 
## 3. Usage
- Download seqs [(Visual BenchMark)](http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html) and save to Sequence. run KCF with CNN features. Choose to run different CNN of single conv-layer or coarse-to-fine features in run.py.

  ```
  python run.py
  ```
- Test old vision of KCF with hog features in 2015-KCF-DSST.
  ```
  python run.py
  ```
