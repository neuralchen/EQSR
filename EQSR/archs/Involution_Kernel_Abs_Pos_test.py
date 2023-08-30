#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: Involution.py
# Created Date: Tuesday July 20th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 30th December 2021 1:11:16 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


from math import sqrt
from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn


from collections import namedtuple
import cupy
from string import Template

from einops import rearrange
import numbers
import math
import matplotlib.pyplot as plt
import os

import numpy as np

scale1 = 4.25
grid = 3
kernel_dir = './hat/archs/kernel/'
os.makedirs(kernel_dir, exist_ok=True)
NK_num = 0
mean_path = kernel_dir + "kernel_mean_" + str(scale1) + ".png"
var_path = kernel_dir + "kernel_var_" + str(scale1) + ".png"

def large_array(a, grid):
    b = np.repeat(a, grid, axis=0)
    b = np.repeat(b, grid, axis=1)
    return b

def view_kernel(y):
    print(y.shape)
    global grid
    global mean_path
    global var_path
    y_mean = y.mean(dim = 0)
    y_mean = y_mean.mean(dim = 0).mean(dim = -1).mean(dim = -1)
    y_var = y.var(dim=0)
    y_var = y_var.var(dim=0).var(dim = -1).var(dim=-1)
    y_mean = y_mean.squeeze().detach().cpu().numpy()
    y_var = y_var.squeeze().detach().cpu().numpy()
    print('y_mean', y_mean.shape)
    print(y_mean)
    y_mean = large_array(y_mean, grid)
    y_var = large_array(y_var, grid)
    print(y_mean.min(), y_mean.max())
    plt.imsave(mean_path, y_mean, cmap="jet")
    plt.imsave(var_path, y_var, cmap="jet")

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_involution_kernel = kernel_loop + '''
extern "C"
__global__ void involution_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h)
            * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset];
        }
      }
    }
    top_data[index] = value;
  }
}
'''


_involution_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void involution_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            value += weight_data[offset_weight] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''


_involution_kernel_backward_grad_weight = kernel_loop + '''
extern "C"
__global__ void involution_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${num};
      ${Dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
              * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
              * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''


class _involution(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)

        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('involution_forward_kernel', _involution_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, groups=weight.size()[1],
                            bottom_height=height, bottom_width=width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda and grad_output.is_contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels, groups=weight.size()[1],
                   bottom_height=height, bottom_width=width,
                   top_height=output_h, top_width=output_w,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('involution_backward_grad_input_kernel',
                                _involution_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())

                n = grad_weight.numel()
                opt['nthreads'] = n

                f = load_kernel('involution_backward_grad_weight_kernel',
                                _involution_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_weight, None, None, None
 

def _involution_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ involution kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2)//stride == weight.size(-2)
    assert input.size(-1)//stride == weight.size(-1)
    if input.is_cuda:
        out = _involution.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out += bias.view(1,-1,1,1)
    else:
        raise NotImplementedError
    return out


class involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 8
        self.groups = self.channels // self.group_channels
        self.seblock = nn.Sequential(
            nn.Conv2d(in_channels = channels, out_channels = channels // reduction_ratio, kernel_size= 3, padding=1),
            # nn.InstanceNorm2d(channels // reduction_ratio, affine=True, momentum=0),
            # nn.BatchNorm2d(channels // reduction_ratio, affine=True, momentum=0), 
            nn.ReLU(),
            nn.Conv2d(in_channels = channels // reduction_ratio, out_channels = kernel_size**2 * self.groups, kernel_size= 1)
        )

        # self.conv1 = ConvModule(
        #     in_channels=channels,
        #     out_channels=channels // reduction_ratio,
        #     kernel_size=1,
        #     conv_cfg=None,
        #     norm_cfg=dict(type='BN'),
        #     act_cfg=dict(type='ReLU'))
        # self.conv2 = ConvModule(
        #     in_channels=channels // reduction_ratio,
        #     out_channels=kernel_size**2 * self.groups,
        #     kernel_size=1,
        #     stride=1,
        #     conv_cfg=None,
        #     norm_cfg=None,
        #     act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

    def forward(self, x, factor):
        # weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        weight = self.seblock(x)
        b, c, h, w = weight.shape
        # factor = factor.expand(b,self.groups,self.kernel_size,self.kernel_size,h,w)
        # print("factor shape:",factor.shape)
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w) * factor
        view_kernel(weight)

                      
        out = _involution_cuda(x, weight, stride=self.stride, padding=(self.kernel_size-1)//2)
        return out

class involution_mod(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride=1):
        super(involution_mod, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 8
        self.groups = self.channels // self.group_channels
        self.seblock = nn.Sequential(
            nn.Conv2d(in_channels = channels, out_channels = channels // reduction_ratio, kernel_size= 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = channels // reduction_ratio, out_channels = kernel_size**2 * self.groups, kernel_size= 1)
        )
        self.lin = nn.Linear(81, kernel_size**2 * self.groups)

        # self.conv1 = ConvModule(
        #     in_channels=channels,
        #     out_channels=channels // reduction_ratio,
        #     kernel_size=1,
        #     conv_cfg=None,
        #     norm_cfg=dict(type='BN'),
        #     act_cfg=dict(type='ReLU'))
        # self.conv2 = ConvModule(
        #     in_channels=channels // reduction_ratio,
        #     out_channels=kernel_size**2 * self.groups,
        #     kernel_size=1,
        #     stride=1,
        #     conv_cfg=None,
        #     norm_cfg=None,
        #     act_cfg=None)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

    def forward(self, x, sin_cos):
        # weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        weight = self.seblock(x)
        r_weight = self.lin(sin_cos).unsqueeze(-1).unsqueeze(-1)
        weight = weight*r_weight
        b, c, h, w = weight.shape
        # factor = factor.expand(b,self.groups,self.kernel_size,self.kernel_size,h,w)
        # print("factor shape:",factor.shape)
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
                      
        out = _involution_cuda(x, weight, stride=self.stride, padding=(self.kernel_size-1)//2)
        return out

class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel): 
        super(ECA_layer, self).__init__()

        b = 1
        gamma = 2
        k_size = int(abs(math.log(channel,2)+b)/gamma)
        k_size = k_size if k_size % 2 else k_size+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, use_CA):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.use_CA = use_CA
        
        self.conv1 = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(hidden_features*2, dim, kernel_size=1, bias=True)
        # if act == "GELU":
        #     self.act = nn.GELU()
        # elif act == "ReLU":
        #     self.sct = nn.ReLU()
        # elif act == "LeakyReLU":
        #     self.act = nn.LeakyReLU()
        # else:
        #    raise Exception("Act layer not implement")    

        if use_CA:
            self.CA = ECA_layer(dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        if self.use_CA:
            x = self.CA(x)
        return x

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Invo_TransformerBlock(nn.Module):
    def __init__(self, dim, kernel_size, stride, ffn_expansion_factor, LayerNorm_type, use_CA, attn_type, ffd_type="ours"):
        super(Invo_TransformerBlock, self).__init__()

        # print("TransFormerBlock type -- ", attn_type, ffd_type)     

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        if attn_type == "Invo":
            self.attn = involution(dim, kernel_size, stride)
        else:
            raise Exception("self.attn not implement")

        self.norm2 = LayerNorm(dim, LayerNorm_type)

        if ffd_type == "ours":
            self.ffn = FeedForward(dim, ffn_expansion_factor, use_CA=use_CA)
        else:
            raise Exception("self.ffn not implement")

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x, factor):
        xn1 = self.norm1(x)
        xn1 = self.conv1(xn1)
        xn1 = self.attn(xn1, factor)
        xn1 = self.conv2(xn1)
        x = x + xn1
        x = x + self.ffn(self.norm2(x))
        return x