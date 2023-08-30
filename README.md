# Deep Arbitrary-Scale Image Super-Resolution via Scale-Equivariance Pursuit (EQSR)
## Accepted by CVPR2023
**The official repository with Pytorch**

Our paper can be downloaded from [EQSR](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Deep_Arbitrary-Scale_Image_Super-Resolution_via_Scale-Equivariance_Pursuit_CVPR_2023_paper.pdf).

## Introduction

EQSR is designed to pursue scale-equivariance image super-resolution.
We compare the PSNR degradation rate of our method and ArbSR.
Taking the SOTA fixed-scale method HAT as reference, our model presents a more stable degradation as the scale increases, reflecting
the equivariance of our method.
![motivation](/doc/img/motivation.PNG)


## Installation
**Clone this repo:**
```bash
git clone https://github.com/neuralchen/EQSR.git
cd EQSR
```
**Dependencies:**
- python3.7+
- pytorch
- pyyaml, scipy, tqdm, imageio, einops, opencv-python
- cupy

(Note: Please do not directly use "pip install" to install basicsr. It might lead to some issues due to version differences.)

## Training

We divide the training into two stages. The first stage involves pretraining on the ImageNet dataset, and the second stage entails fine-tuning on the DF2K dataset.

You can modify parameters such as batch size, iterations, learning rate, etc. in the configuration files.

-  Phase 1
Modify the dataset path in options/train/train*.xml, and run the following command to train.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4320 train.py -opt train_EQSR_ImageNet_from_scratch --launcher pytorch
```
-  Phase 2
Modify the paths of the datasets and the location of the pretrained model in the configuration file.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4320 train.py -opt train_EQSR_finetune_from_ImageNet_pretrain --launcher pytorch
```

## Datasets
TODO

### Preprocess
TODO

## Inference with a pretrained EQSR model
### Pretrained Models
- Baidu Netdisk (百度网盘)：https://pan.baidu.com/s/1ui-GSbAQLuTyOmxBlAQZVg 
- Extraction Code (提取码)：lspg

Modify the dataset path and pre-trained model path in options/test/test*.xml, and run the following command to test.
If GPU memory is limited, you can consider adding ```PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32``` before these commands.

```
python test.py -opt options/test/testx234.yml
python test.py -opt options/test/testx6.yml
python test.py -opt options/test/testx8.yml
```

## Ackownledgements
This code is built based on [HAT](https://github.com/XPixelGroup/HAT) and [BasicSR](https://github.com/XPixelGroup/BasicSR). We thank the authors for sharing the codes.

## To cite our paper

## Related Projects
