#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: data_loader_file.py
# Created Date: Saturday April 4th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 21st December 2021 12:52:25 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import os
import cv2
# cv2.setNumThreads(1)

import glob
import math
import random
import numpy as np
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from basicsr.data.data_util import scandir
import pickle

import torch
from torch.utils import data

from basicsr.utils.registry import DATASET_REGISTRY

def make_coord(shape):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        # v0, v1 = -1, 1

        r = 1 / n
        seq = -1 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    # ret = torch.stack(torch.meshgrid(coord_seqs, indexing='ij'), dim=-1)
    ret = torch.stack(torch.meshgrid(coord_seqs), dim=-1)
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    # rgb = img.view(3, -1).permute(1, 0)
    return coord


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.__preload__()

    def __preload__(self):
        try:
            # self.hr, self.lr = next(self.dataiter)
            self.lr, self.hr, self.coord, self.cell = next(self.dataiter)
        except StopIteration:
            # self.loader.dataset.shuffle()
            self.dataiter = iter(self.loader)
            self.lr, self.hr, self.coord, self.cell = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.hr     = self.hr.cuda(non_blocking=True)
            self.hr     = (self.hr - 0.5) * 2.0
            self.lr     = self.lr.cuda(non_blocking=True)
            self.lr     = (self.lr - 0.5) * 2.0
            self.coord  = self.coord.cuda(non_blocking=True)
            self.cell   = self.cell.cuda(non_blocking=True)
            # self.hr     = (self.hr/255.0 - 0.5) * 2.0
            # self.lr     = (self.lr/255.0 - 0.5) * 2.0
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        hr     = self.hr
        lr     = self.lr
        coord  = self.coord
        cell   = self.cell
        self.__preload__()
        
        return lr, hr, coord, cell
    
    def __len__(self):
        """Return the number of images."""
        return len(self.loader)

@DATASET_REGISTRY.register()
class DSF_imagenet_Dataset(data.Dataset):
    def __init__(  
                    self,
                    opt
                ):
        super(DSF_imagenet_Dataset, self).__init__()
        """Initialize and preprocess the flickr and div2k dataset."""

        div2k_root = opt['dataroot_gt']
        # div2k_root="/data3/KITTI/DF2K_HR"
        # div2k_root="/data3/GeoSR_data/benchmark/Set5/HR"
        lr_patch_size=opt["patch_size"]  #48
        degradation = "bicubic"
        min_scale = 1
        max_scale = 4
        subffix='png'
        random_seed=1234
        dataset_enlarge=opt["dataset_enlarge_ratio"]  #20

        self.div2k_root     = div2k_root
        self.degradation    = degradation
        self.min_scale      = min_scale
        self.max_scale      = max_scale
        self.l_ps           = lr_patch_size
        self.d_e            = dataset_enlarge
        self.subffix        = subffix
        self.dataset        = []
        self.random_seed    = random_seed
        random.seed(self.random_seed)
        self.__preprocess__()
        self.num_images = len(self.data_path_list)

    def __preprocess__(self):
        """Preprocess the Artworks dataset."""
        
        data_path_list = []
        # div2khr_path  = os.path.join(self.div2k_root,"DIV2K_train_HR")#.replace('/','_'))
        div2khr_path  = self.div2k_root

        print("processing DIV2K images...")
        """temp_path   = os.path.join(div2khr_path,'*.%s'%(self.subffix))
        images      = glob.glob(temp_path)
        for item in images:
            data_path_list.append(item)"""

        data_path_list = sorted(list(scandir(div2khr_path, suffix="JPEG", recursive=True, full_path=True)))
        
        # random.shuffle(data_path_list)

        # self.dataset = images
        print('Finished preprocessing the DIV2K dataset, total image number: %d...'%len(data_path_list))

        self.data_path_list = data_path_list
        if not os.path.exists('./imagenet_list.pkl'):

            for item in tqdm(data_path_list[:]):
                hr_img      = cv2.imread(item)
                hr_img      = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
                # hr_img      = hr_img.transpose((2, 0, 1))
                # hr_img      = torch.from_numpy(hr_img)
                if hr_img.shape[0] <= (self.max_scale*self.l_ps+1) or hr_img.shape[1] <= (self.max_scale*self.l_ps+1):
                    self.data_path_list.remove(item)
                
            #     self.dataset.append(hr_img)
            # self.indeices    = [i for i in range(len(self.dataset))]
            # self.indeices    = self.indeices * self.d_e
        
            tmp_file = open("./imagenet_list.pkl", 'wb')
            pickle.dump(self.data_path_list, tmp_file)
            tmp_file.close()
            
        else:
            print("min pixels:", self.max_scale*self.l_ps)
            self.data_path_list = pickle.load(open( './imagenet_list.pkl', 'rb'))
            if not os.path.exists(self.data_path_list[0]):
                print(self.data_path_list[0])
                for i, item in enumerate(self.data_path_list):
                    sp = item.split('/')
                    cls_name, pic_name = sp[-2], sp[-1]
                    self.data_path_list[i] = os.path.join(self.div2k_root, cls_name, pic_name)



        print("----->", len(self.data_path_list))
        # random.shuffle(self.indeices)
        # indeices = np.random.randint(0,len(self.dataset),size=self.d_e*len(self.dataset))
        # self.pathes= indeices.tolist()
        print("Finish to read the dataset!")
        # import sys
        # mem_ocu = sys.getsizeof(self.dataset) / 1024.0 / 1024.0 / 1024.0
        # print("Dataset Memory occupy: %.2f GB"%mem_ocu)
    # def shuffle(self):
    #     print("Reshuffle the dataset")
    #     # self.indeices = np.random.randint(0,len(self.dataset),size=self.d_e*len(self.dataset))
    #     random.shuffle(self.indeices)
    
    def resize_fn(self, img, size):
        return transforms.ToTensor()(
            transforms.Resize(size, InterpolationMode.BICUBIC)(
                transforms.ToPILImage()(img)))
    
    # def resize_fn(self, img, size):
    #     return transforms.Resize(size, Image.BICUBIC)(img)

    def __getitem__(self, index):
        """Return one hr image and its corresponding lr image."""
        # hr_img  = self.dataset[self.indeices[index]]
        # hr_img  = self.indeices[index]
        
        hr_img      = cv2.imread(self.data_path_list[index])
        hr_img      = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
        hr_img      = hr_img.transpose((2, 0, 1))
        hr_img      = torch.from_numpy(hr_img)
        # print(self.data_path_list[index], hr_img.shape)
        
        s       = random.uniform(self.min_scale, self.max_scale)
        w_lr    = self.l_ps
        w_hr    = round(w_lr * s)

        # if hr_img.shape[-2]<w_hr or hr_img.shape[-1]<w_hr:
        #     print("---->", self.data_path_list[index], hr_img.shape, w_hr)
        # print(hr_img.shape[-2] - w_hr, hr_img.shape[-1] - w_hr)
        
        x0      = random.randint(0, hr_img.shape[-2] - w_hr)
        y0      = random.randint(0, hr_img.shape[-1] - w_hr)
        crop_hr = hr_img[:, x0: x0 + w_hr, y0: y0 + w_hr]
        # crop_hr = (crop_hr / 255.0 - 0.5) * 2.0
        crop_hr = crop_hr / 255.0
        # crop_lr = F.interpolate(crop_hr.unsqueeze(0), w_lr, mode=self.degradation, align_corners=True)
        crop_lr = self.resize_fn(crop_hr, (w_lr, w_lr))
        
        # crop_lr = crop_lr[0,:,:,:]

        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        dflip = random.random() < 0.5

        def augment(x):
            if hflip:
                x = x.flip(-2)
            if vflip:
                x = x.flip(-1)
            if dflip:
                x = x.transpose(-2, -1)
            return x

        crop_lr = augment(crop_lr)
        crop_hr = augment(crop_hr)

        # flip_ran= random.randint(0,2)
        # if flip_ran == 0:
        #     # horizontal
        #     crop_hr  = torch.flip(crop_hr,[1])
        #     crop_lr  = torch.flip(crop_lr,[1])
        # elif flip_ran == 1:
        #     # vertical
        #     crop_hr  = torch.flip(crop_hr,[2])
        #     crop_lr  = torch.flip(crop_lr,[2])
        
        # rot_ran = random.randint(0,3)
        # if rot_ran != 0:
        #     # horizontal
        #     crop_hr  = torch.rot90(crop_hr, rot_ran, [1, 2])
        #     crop_lr  = torch.rot90(crop_lr, rot_ran, [1, 2])

        hr_coord    = make_coord(crop_hr.shape[-2:])

        x0          = random.randint(0, w_hr - w_lr)
        y0          = random.randint(0, w_hr - w_lr)
        hr_coord    = hr_coord[x0: x0 + w_lr, y0: y0 + w_lr,:]
        crop_hr     = crop_hr[:, x0: x0 + w_lr, y0: y0 + w_lr]
 
        cell = torch.ones_like(hr_coord)
        cell *= 2 / w_hr
        # cell[:,:, 1] *= 2 / w_hr

        # cell *= w_lr
        cell = cell.permute(2, 0, 1)

        # return crop_lr, crop_hr, hr_coord, cell
        return {'lq': crop_lr, 'gt': crop_hr, 'gt_path': self.div2k_root, 'coord':hr_coord, "cell":cell}
        
    def __len__(self):
        """Return the number of images."""
        return self.num_images

def GetLoader(  dataset_roots,
                batch_size=16,
                **kwargs
            ):
    """Build and return a data loader."""
    if not kwargs:
        a = "Input params error!"
        raise ValueError(print(a))

    random_seed       = kwargs["random_seed"]
    num_workers       = kwargs["dataloader_workers"]
    degradation       = kwargs["degradation"]
    min_scale         = kwargs["min_scale"]
    max_scale         = kwargs["max_scale"]
    lr_patch_size     = kwargs["lr_patch_size"]
    subffix           = kwargs["subffix"]
    div2k_root        = dataset_roots
    dataset_enlarge   = kwargs["dataset_enlarge"]
    

    content_dataset = DIV2K_Dataset(
                        div2k_root,
                        lr_patch_size,
                        degradation,
                        min_scale,
                        max_scale,
                        subffix,
                        random_seed,
                        dataset_enlarge
                    )

    content_data_loader = data.DataLoader(
                        dataset=content_dataset,
                        batch_size=batch_size,
                        drop_last=True,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True
                    )
                        
    prefetcher = DataPrefetcher(content_data_loader)
    # return content_data_loader
    return prefetcher

if __name__ == "__main__":
    

    dataset_path = {
        "flickr":"F:\\DataSet\\Flickr2K",
        "div2k":"F:\\DataSet\\DIV2K"
    }

    memory_dataloader = GetLoader(dataset_path["div2k"],
        16,
        1234,
        **{
        "dataloader_workers": 4,
        "subffix": "png",
        "lr_patch_size": 48,
        "degradation": "bicubic",
        "dataset_enlarge": 5,
        "min_scale": 1,
        "max_scale": 4
    })
    # hdf5_dataloader = iter(hdf5_dataloader)
    import time
    import datetime
    import cv2
    start_time = time.time()

    def tensor2img(img_tensor):
        res = img_tensor.numpy()
        res = (res + 1) / 2
        res = np.clip(res, 0.0, 1.0)
        res = res * 255
        res = res.transpose((0,2,3,1))
        return res
    
    # dataiter = iter(memory_dataloader)

    for i in range(1000):
        lr, hr, coord, cell = memory_dataloader.next()

        # try:
        #     # self.hr, self.lr = next(self.dataiter)
        #     lr, hr, coord, cell = next(dataiter)
        # except StopIteration:
        #     dataiter = iter(memory_dataloader)
        #     lr, hr, coord, cell = next(dataiter)
        # lr,hr = next(hdf5_dataloader)
        hr = hr.cpu()
        res = tensor2img(hr).astype(np.uint8)
        hr = cv2.cvtColor(res[0,:,:,:], cv2.COLOR_RGB2BGR)
        cv2.imwrite("wocaohr_%d.png"%i,hr)

        lr = lr.cpu()
        res = tensor2img(lr).astype(np.uint8)
        lr = cv2.cvtColor(res[0,:,:,:], cv2.COLOR_RGB2BGR)
        cv2.imwrite("wocaolr_%d.png"%i,lr)
        # hr = hr +1
    elapsed = time.time() - start_time
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Elapsed [{}]".format(elapsed))
