# VAFF-Net
This repository holds the Pytorch implementation of VAFF-Net. 

## Introduction
We propose a Voting-based Adaptive Feature Fusing Multi-task Network (VAFF-Net) for joint learning of retinal vessel (RV), foveal avascular zone (FAZ), and retinal vascular junction (RVJ) in OCTA images. In addition, our proposed method can be used as a general  multi-task learning framework, and We validate it on the public DRIVE dataset.

## Getting Started

Clone this repo
```
git clone https://github.com/iMED-Lab/VAFF-Net.git
```

Install prerequisites
```
cd VAFF-Net
pip install -r requirements.txt
```

Prepare your data

Please put the root directory of your dataset into the folder ./data. The root directory contain the two subfolder now: ROSE-MT (the public OCTA dataset with multi-task annotations), DRIVE-MT (the public fundus dataset with multi-task annotations). 

You can change the path of the dataset and other configurations in the ./config.py 

The information about the ROSE dataset with multi-task annotations could be seen in the following link: 

https://imed.nimte.ac.cn/ROSE-O.html

## Running the code

### Start Visdom
```
    python -m visdom.server  -p 2333
```
### Training on the OCTA dataset
```
    python train-OCTA.py
```
### Training on the DRIVR dataset
```
    python train-DRIVE.py
```

<span id="jump2"></span>
### Citation
If you use this code for your research, please cite our papers. 
```
@article{hao2022vaffnet,
  title={Retinal Structure Detection in OCTA Image via Voting-based Multi-task Learning},
  author={Jinkui Hao, Yonghuai Liu, Ardhendu Behera, Jiong Zhang, Yitian Zhao},
  year={2022},
}
```
