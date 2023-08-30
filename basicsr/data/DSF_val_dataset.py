import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import pickle
import os
import imageio
import json

from basicsr.utils.registry import DATASET_REGISTRY

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

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

class ImageFolder(data.Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x

class PairedImageFolders(data.Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]

@DATASET_REGISTRY.register()
class DSF_val_Dataset(data.Dataset):

    def __init__(self, opt):
        super(DSF_val_Dataset, self).__init__()
        # inp_size=None, augment=False, sample_q=None  
        # root1 = "/data3/GeoSR_data/benchmark/Set5/LR_bicubic/X4"
        # root2 = "/data3/GeoSR_data/benchmark/Set5/HR"

        root1=opt['dataroot_lq']
        root2=opt['dataroot_gt']

        dataset = PairedImageFolders(root1, root2)
        self.dataset = dataset
        self.inp_size = None
        self.augment = False
        self.sample_q = None
        self.opt = opt
        self.root = root2
        print("---> DSF val Dataset")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        # if self.inp_size is None:
        h_lr, w_lr = img_lr.shape[-2:]
        img_hr = img_hr[:, :h_lr * s, :w_lr * s]
        crop_lr, crop_hr = img_lr, img_hr

        hr_coord = to_pixel_samples(crop_hr.contiguous())

        cell = torch.ones_like(hr_coord)
        cell[:, :, 0] = cell[:, :, 0] * 2 / crop_hr.shape[-2]
        cell[:, :, 1] = cell[:, :, 1] * 2 / crop_hr.shape[-1]
        cell = cell.permute(2, 0, 1)

        # cell = torch.ones_like(hr_coord)
        # cell *= 2 / w_hr
        # cell[:,:, 1] *= 2 / w_hr

        # cell = cell.permute(2, 0, 1)

        # print("crop_lr--", crop_lr.shape)
        # print("crop_hr--", crop_hr.shape)
        # print("coord--", hr_coord.shape)
        # print("cell--", cell.shape)

        return {'lq': crop_lr, 'gt': crop_hr, 'gt_path': self.root, 'coord':hr_coord, "cell":cell}

@DATASET_REGISTRY.register()
class DSF_val_downsample_Dataset(data.Dataset):
    def __init__(self, opt):
        super(DSF_val_downsample_Dataset, self).__init__()
        root = opt['dataroot_gt']
        dataset = ImageFolder(root)
        inp_size=None
        scale_min=opt['scale']
        scale_max=None
        augment=False
        sample_q=None
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.root = root
        self.opt = opt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        hr_coord= to_pixel_samples(crop_hr.contiguous())

        cell = torch.ones_like(hr_coord)
        cell[:, :, 0] = cell[:, :, 0] * 2 / crop_hr.shape[-2]
        cell[:, :, 1] = cell[:, :, 1] * 2 / crop_hr.shape[-1]
        cell = cell.permute(2, 0, 1)

        # cell = torch.ones_like(hr_coord)
        # cell *= 2 / w_hr
        # cell = cell.permute(2, 0, 1)

        # cell = torch.ones_like(hr_coord)
        # print("cell shape:",cell.shape)
        # print("crop_hr shape:",crop_hr.shape)
        # cell[:,:, 0] = cell[:,:, 0] *2 / crop_hr.shape[-2] 
        # cell[:,:, 1] = cell[:,:, 1] *2 / crop_hr.shape[-1]
        # cell[:,:, 0] = cell[:,:, 0] *0
        # cell[:,:, 1] = cell[:,:, 1] *0
        # print("hello")
        # if self.inp_size is None:
        #     cell[:,:, 0] *= h_lr
        #     cell[:,:, 1] *= w_lr
        # else:
        #     cell *= self.inp_size
        # cell = cell.permute(2, 0, 1)

        return {'lq': crop_lr, 'gt': crop_hr, 'gt_path': self.root, 'coord':hr_coord, "cell":cell}

