# Deep Arbitrary-Scale Image Super-Resolution via Scale-Equivariance Pursuit (EQSR)
## Accepted by CVPR2023
**The official repository with Pytorch**

Our paper can be downloaded from [[Arxiv]]().
Try EQSR in Colab [ <a href="https://colab.research.google.com/github/neuralchen/EQSR/blob/main/demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/github/neuralchen/EQSR/blob/main/demo.ipynb)

## Introduction

EQSR is designed to pursue scale-equivariance image super-resolution.
We compare the PSNR degradation rate of our method and ArbSR.
Taking the SOTA fixed-scale method HAT as reference, our model presents a more stable degradation as the scale increases, reflecting
the equivariance of our method.
![motivation](/doc/img/motivation.PNG)

## Attention

***We are archiving our code and awaiting approval for code public access!***

***The code will be open source within four days, please be patient and enthusiastic***

## Installation
**Clone this repo:**
```bash
git clone https://github.com/neuralchen/EQSR.git
cd EQSR
```
**Dependencies:**
- PyTorch 1.7.0
- Pillow 8.3.1; Matplotlib 3.3.4; opencv-python 4.5.3; Faiss 1.7.1; tqdm 4.61.2; Ninja 1.10.2

All dependencies for defining the environment are provided in `environment/eqsr_env.yaml`.
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/) (you may need to modify `eqsr_env.yaml` to install PyTorch that matches your own CUDA version following [https://pytorch.org/](https://pytorch.org/)):
```bash
conda env create -f ./environment/eqsr_env.yaml
```

## Training

## Inference with a pretrained EQSR model

## Results

![peformance](/doc/img/peformance.PNG)

## To cite our paper

## Related Projects
