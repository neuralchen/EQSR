# Deep Arbitrary-Scale Image Super-Resolution via Scale-Equivariance Pursuit (EQSR)
## Accepted by CVPR2023
**The official repository with Pytorch**

Our paper can be downloaded from [[Arxiv]](). Try EQSR in Colab [ <a href="https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/train.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/github/neuralchen/SimSwap/blob/main/train.ipynb)

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
