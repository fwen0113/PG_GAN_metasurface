import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
import scipy.stats as st
from typing import Tuple
import math
import logging
import matplotlib.pyplot as plt
import scipy.io as io
import os
import codecs, json 
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    

def index_along(tensor, key, axis):
    indexer = [slice(None)] * len(tensor.shape)
    indexer[axis] = key
    return tensor[tuple(indexer)]


def pad_reflect(inputs, padding: int, axis: int):
    if padding % 2 != 0:
        raise ValueError('cannot do centered padding if padding is not even')
    ndim = len(inputs.shape)
    
    if axis < 0:
        axis += ndim
    axis = ndim - axis - 1
    
    if axis == 0:
        paddings = (padding
         // 2, padding // 2, 0, 0)
    else:
        paddings = ((0, 0) * axis +
                  (padding // 2, padding // 2))

    return F.pad(inputs, paddings, mode='reflect')


def pad_antireflect(inputs, padding: int, axis: int):
    if padding % 2 != 0:
        raise ValueError('cannot do centered padding if padding is not even')
    ndim = len(inputs.shape)
    
    if axis < 0:
        axis += ndim
    axis = ndim - axis - 1
    
    if axis == 0:
        paddings = (padding // 2, padding // 2, 0, 0)
    else:
        paddings = ((0, 0) * axis +
                  (padding // 2, padding // 2))

    PAD1 = F.pad(-inputs, paddings, mode='reflect');

    PAD2 = F.pad(inputs, paddings, mode='constant');

    return PAD1 + 2*PAD2



def pad_periodic(inputs, padding: int, axis: int, center: bool = True):
    if center:
        if padding % 2 != 0:
            raise ValueError('cannot do centered padding if padding is not even')
        inputs_list = [index_along(inputs, slice(-padding//2, None), axis),
                       inputs,
                       index_along(inputs, slice(None, padding//2), axis)]
    else:
        inputs_list = [inputs, index_along(inputs, slice(None, padding), axis)]
    return torch.cat(inputs_list, dim=axis)


def pad2d_meta(inputs, padding: Tuple[int, int]):
    padding_y, padding_x = padding
    return pad_periodic(pad_reflect(inputs, padding_y, axis=-2),
                                            padding_x, axis=-1, center=True)

def pad2d_fourier(inputs, padding: Tuple[int, int]):
    """
    Args:
        inputs: B x 2 x H x W
        even real and odd imag x
    """
    padding_y, padding_x = padding
    inputs_real = inputs[:,0,:,:].unsqueeze(1)
    inputs_imag = inputs[:,1,:,:].unsqueeze(1)
    inputs_real = pad_reflect(pad_reflect(inputs_real, padding_y, axis=-2),padding_x, axis=-1)
    inputs_imag = pad_antireflect(pad_reflect(inputs_imag, padding_y, axis=-2),padding_x, axis=-1)
    return torch.cat((inputs_real, inputs_imag), 1)


def gkern(kernlen=7, nsig=4):
    """Returns a 2D Gaussian kernel array."""

    x_cord = torch.arange(kernlen)
    x_cord = torch.arange(0,kernlen)
    x_grid = x_cord.repeat(kernlen).view(kernlen, kernlen)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).type(Tensor)

    mean = (kernlen - 1)/2.
    variance = nsig**2.

    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                  torch.exp(
                      -torch.sum((xy_grid - mean)**2., dim=-1) /\
                      (2*variance)
                  )
    #print('shape is')
    #print(gaussian_kernel)
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    #print('sum')
    #print(torch.sum(gaussian_kernel))

    return gaussian_kernel.type(Tensor).requires_grad_(False)

def conv2d(inputs, kernel, padding='same'):
    """
    Args:
        inputs: B x C x H x W
        gkernel: 2d kernel
    """

    B, C, _, _ = inputs.size()
    kH, kW = kernel.size()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, C, 1, 1)


    
    if padding == 'valid':
        a=F.conv2d(inputs, kernel)
        return F.conv2d(inputs, kernel)
    elif padding == 'same':
        pad = ((kH-1)//2, (kW-1)//2)
   
    

    return F.conv2d(inputs, kernel, padding = pad)


def conv2d_meta(inputs, kernel):
    """
    Args:
        inputs: B x C x H x W
        gkernel: 2d kernel
    """
    
    kH, kW = kernel.size()
    padded_inputs = pad2d_meta(inputs,(kH-1, kW-1))

    return conv2d(padded_inputs, kernel, padding='valid')

def conv2d_fourier(inputs, kernel):
    """
    Args:
        inputs: B x C x H x W
        gkernel: 2d kernel
    """
    kH, kW = kernel.size()
    padded_inputs = pad2d_fourier(inputs,(kH-1, kW-1))
    
    return conv2d(padded_inputs, kernel, padding='valid')

def density_filter(inputs, R):
    """
    Args:
        inputs: B x C x H x W
        R:  radius of density filter
    """
    
    x = torch.linspace(-R, R, int(2*R+1))
    [X1, X2] = torch.meshgrid([x, x])
    #print('test')
    #print(torch.sqrt(X1**2 + X2**2))
    kernel_raw = R - torch.sqrt(X1**2 + X2**2)
    kernel_raw[kernel_raw < 0] = 0
    kernel_raw /= kernel_raw.sum()
    kernel = kernel_raw.type(Tensor).requires_grad_(False)

    return conv2d_meta(inputs, kernel)


def thresh_filter(inputs, B, N):
    #print(N)
    B = 100
    print('N is',N)
    inputs1=inputs.detach().numpy().copy()
    print('input max',np.max(inputs1))
    #temp=temp.type(Tensor)
    #inputs=inputs.type(Tensor)
    inputs2=inputs1.copy()
    threshold1=(inputs1 < N).copy()
    threshold2=(inputs2 > N).copy()

    input_shape=np.shape(inputs1)
    temp1 = inputs1[threshold1].copy()
    temp2 = inputs2[threshold2].copy()
    #threshold=threshold.type(torch.LongTensor)
    #threshold1=threshold1.type(torch.LongTensor)
    #threshold=threshold.type(Tensor)
    #threshold1=threshold1.type(Tensor)
    #inputs1[threshold1] = N*torch.exp(-B*(1- temp/N))
    inputs1[threshold1] = N*np.exp(-B*(1- temp1/N)) - (N - temp1) * np.exp(-B)

    #inputs1[threshold1] = 0
    #temp2=temp1.type(Tensor)
    inputs2[threshold2] = 1.0 - (1.0-N)*np.exp(-B*(temp2 - N)/(1.0-N)) - (N - temp2) * np.exp(-B)
    #inputs2[threshold2] = 1.0 
    #input_final=inputs1*threshold1 + inputs2*threshold2
    #print("temp size",np.shape(temp2.reshape(input_shape)))
    input_final=inputs1*threshold1 + inputs2*threshold2
    print('max',np.max(inputs1))

    input_final = torch.tensor(input_final)
    #input_final=input_final.type(Tensor)
    #print(threshold+threshold1)
    #print(inputs[inputs > N])
    #print(inputs)
    return input_final



class ConvTranspose2d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 groups=1, bias=True, dilation=1):
        super().__init__()
        self.padding = kernel_size - 1
        self.trim = self.padding * stride // 2
        pad = (kernel_size - stride) // 2 
        self.output_padding = (kernel_size - stride) % 2 
        self.conv2d_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=pad,
                                          output_padding=0, groups=groups, bias=bias, dilation=dilation)
    
    def forward(self, inputs):
        padded_inputs = pad2d_meta(inputs, (self.padding, self.padding))
        padded_outputs = self.conv2d_transpose(padded_inputs)
        
        if self.output_padding:
            padded_outputs = padded_outputs[:, :, 1:, 1:]
            
        if self.trim:
            return padded_outputs[:, :, self.trim:-self.trim, self.trim:-self.trim]
        else:
            return padded_outputs
 
    
class Conv2d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1)*dilation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, 
                                dilation = dilation, groups = groups, bias = bias)
    
    def forward(self, inputs):
        padded_inputs = pad2d_meta(inputs, (self.padding, self.padding))
        return self.conv2d(padded_inputs) 

class Conv2d_fourier(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1)*dilation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, 
                                dilation = dilation, groups = groups, bias = bias)
    
    def forward(self, inputs):
        padded_inputs = pad2d_fourier(inputs, (self.padding, self.padding))
        return self.conv2d(padded_inputs) 

class Deform2d_meta(nn.Module):
    """docstring for Deform2d_meta"""
    def __init__(self, R, B, N, beam_blur, wspan, wc):
        super().__init__()
        self.R = R
        self.B = B
        self.N = N
        self.beam_blur = beam_blur
       
        self.wspan = wspan.type(Tensor)
        self.wc = wc.type(Tensor)


    def forward(self, inputs, labels):

        dr_nm = (torch.mean(labels[:, 0])*self.wspan+self.wc)/93.0
        blur_grid = torch.floor(self.beam_blur/dr_nm)
        plot_try=inputs
        #print('plot_content')
        content = plot_try.detach().numpy()
        content = content[0,0,:,:]

        b = content.tolist()
        file_path = "before_rand11.json" ## your path variable
        json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format



        filename = 'before.mat'
        file_path = os.path.join(filename)
        
        # nested lists with same data, indices


        gauss_kernel = gkern(6*int(blur_grid.item())+1, blur_grid)



        a_filt = density_filter(inputs, torch.floor(self.R/dr_nm).item()) #convolution

        #a_des = thresh_filter(a_filt, self.B, 0.5)
        a_des = a_filt
      
        a_filt2 = conv2d_meta(a_des, gauss_kernel)

        content = a_filt2[0,0,:,:]
        b = content.tolist()
        file_path = "before_rand.json" ## your path variable
        json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format


        content_e=thresh_filter(a_filt2, self.B, self.N[0])
        content_i=thresh_filter(a_filt2, self.B, self.N[1])
        content_d=thresh_filter(a_filt2, self.B, self.N[2])
        a_e=content_e
        a_i=content_i
        a_d=content_d
        content_e=content_e[0,0,:,:]
        content_i=content_i[0,0,:,:]
        content_d=content_d[0,0,:,:]
        
        print('dim',np.shape(a_e))


        #content_e = content
        #content_i = content
        #content_d = content
        eee = content_e.tolist() # nested lists with same data, indices
        iii = content_i.tolist()
        ddd = content_d.tolist()


        file_path = "after_e.json" ## your path variable
        json.dump(eee, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
        file_path = "after_i.json" ## your path variable
        json.dump(iii, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
        file_path = "after_d.json" ## your path variable
        json.dump(ddd, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
        


        print('done')
        return a_e, a_i, a_d


class FFT(nn.Module):
    """Fourier transform"""
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        # inputs: N x C x H x W
        inputs = inputs[:, 0, :, :]
        inputs = torch.cat((inputs,torch.flip(inputs, [1])), 1)
        FF = torch.rfft(inputs, 2, onesided=False)
        FF = FF[:,:FF.size(1)//2,:FF.size(2)//2, :].permute(0,3,1,2)
        return FF


class IFFT(nn.Module):
    """Fourier transform"""
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1)
        real_FF = inputs[:, :, :, 0].unsqueeze(-1)
        imag_FF = inputs[:, :, :, 1].unsqueeze(-1)
        real_FF = torch.cat((real_FF, torch.flip(real_FF, [1])), 1)
        real_FF = torch.cat((real_FF, torch.flip(real_FF, [2])), 2)
        imag_FF = torch.cat((imag_FF, torch.flip(imag_FF, [1])), 1)
        imag_FF = torch.cat((imag_FF, -torch.flip(imag_FF, [2])), 2)

        FF = torch.cat((real_FF, imag_FF), 3)

        img = torch.ifft(FF, 2)[:,:,:,0].unsqueeze(1)
        img = img[:, :, :img.size(2)//2, :]
        
        return img




        
