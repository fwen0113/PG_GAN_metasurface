import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 
from collections import OrderedDict
from math import ceil


class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dims
        self.label_dim = params.label_dims
        self.size_temp = params.size_temp

        self.gkernel = gkern(params.gkernlen, params.gkernsig)

        self.blocks = nn.ModuleList()

        self.size_temp = params.size_temp
        self.max_res = params.max_res
        

        self.FC = nn.Sequential(
            nn.Linear((self.noise_dim + self.label_dim), 512),
            nn.ReLU(),
            nn.Linear(512, 4*16*self.size_temp, bias=False),
            nn.BatchNorm1d(4*16*self.size_temp),
            nn.ReLU()
        )

        self.blocks.append(nn.Sequential(OrderedDict([
                ('conv0',ConvTranspose2d_meta(self.size_temp,self.size_temp, 5, stride=2, bias=False)),
                ('conv1',nn.BatchNorm2d(self.size_temp)),
                ('conv2',nn.ReLU())
                ])))

        for block_index in range(self.max_res-2):
            n_in = int(self.size_temp/(2**block_index))
            print('n_in',n_in)
            n_out = int(self.size_temp/(2*2**block_index))
            print('n_out',n_out)
            self.blocks.append(nn.Sequential(OrderedDict([
                ('conv0',ConvTranspose2d_meta(n_in, n_out, 5, stride=2, bias=False)),
                ('conv1',nn.BatchNorm2d(n_out)),
                ('conv2',nn.ReLU())
                ])))

        self.to2d = nn.ModuleList()
        self.to2d.append(ConvTranspose2d_meta(self.size_temp, 1,5, stride=1, bias=False))
        for i in range(self.max_res-1):
            # max of nch * 32 feature maps as in the original article (with nch=16, 512 feature maps at max)
            n_in = int(self.size_temp/(2**i))
    
            self.to2d.append(ConvTranspose2d_meta(n_in, 1,5, stride=1, bias=False))

        self.CONV = nn.Sequential(
            ConvTranspose2d_meta(64, 64, 5, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvTranspose2d_meta(64, 32, 5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ConvTranspose2d_meta(32, 16, 5, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ConvTranspose2d_meta(16, 1, 5),
            )


    def forward(self, noise, labels, params):
        net = torch.cat([labels, noise], -1)
        print('begin',net.size())
        net = self.FC(net)
        net = net.view(-1, 64, 4, 16)
        self.scheme_index = params.scheme_index
        y0 = net
        if(self.scheme_index is not 0):
            net = self.blocks[0](net)
        if(self.scheme_index == 2):
            y0 = self.blocks[0](y0)
            
        for i in range(1, self.scheme_index):
            print('block is',self.blocks[i])
            net = self.blocks[i](net)
            if(i == self.scheme_index-2):
                y0 = net

        net = self.to2d[self.scheme_index](net)

        if(self.scheme_index is not 0):
            y0 = self.to2d[self.scheme_index-1](F.upsample(y0, scale_factor=2))
            net = params.alpha * net + (1 - params.alpha) * y0

        return torch.tanh(net)




class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, params):
        super().__init__()
        self.size_temp = params.size_temp

        self.FC = nn.Sequential(
            nn.Linear(4*16*self.size_temp + params.label_dims, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
            )

        self.from2d = nn.ModuleList()

        self.blocks = nn.ModuleList()

        self.max_res = params.max_res

        self.blocks.append(nn.Sequential(OrderedDict([
                ('conv0',Conv2d_meta(self.size_temp,self.size_temp, 5, stride=2, bias=False)),
                ('conv1',nn.LeakyReLU(0.2)),
                ])))
        
        for block_index in range(0,self.max_res-2):
            n_in = int(self.size_temp/(2*2**block_index))
            #print('n_in',n_in)
            n_out = int(self.size_temp/(2**block_index))
            print('n_out',n_out)


            self.blocks.append(nn.Sequential(OrderedDict([
                ('conv0',Conv2d_meta(n_in, n_out, 5, stride=2, bias=False)),
                ('conv1',nn.LeakyReLU(0.2)),
                ])))


        self.from2d.append(Conv2d_meta(1,self.size_temp,5, stride=1, bias=False))
        for i in range(0,self.max_res-1):
            # max of nch * 32 feature maps as in the original article (with nch=16, 512 feature maps at max)
            n_out = int(self.size_temp/(2**(i)))
            self.from2d.append(Conv2d_meta(1,n_out,5,stride=1, bias=False))

        

    def forward(self, img, labels,params):
        net = img + torch.randn(img.size()).type(Tensor)*params.noise_level
        #net = self.CONV(net)
        y0 = net
        self.scheme_index = params.scheme_index
        net = self.from2d[self.scheme_index](net)
        if(self.scheme_index is not 0):

            net = self.blocks[params.scheme_index-1](net)

            
            y0 = self.from2d[self.scheme_index-1](F.adaptive_avg_pool2d(y0,net[0][0].size()))
                

        net = params.alpha * net + (1 - params.alpha) * y0

        for block_index in range(params.scheme_index-2,-1,-1):
          
            net = self.blocks[block_index](net)

        net = net.view(net.size(0), -1)
        net = torch.cat([net, labels], -1)
        net = self.FC(net)
        
        return net



