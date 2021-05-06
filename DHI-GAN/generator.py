'''
author: lyi
date 20210129
desc generator model, from embedding and noise to dental images


'''

import torch
from torch import nn


import torch.nn.functional as F
from msse import SCALayer
class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.depthwise_conv_bn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels))

    def forward(self, x):
        funnel_x = self.depthwise_conv_bn(x)
        return torch.max(x, funnel_x)











class TI_GenEmd_DS(nn.Module):

    def __init__(self, input_dim=100, embedding_dim = 10, hidden_dim=64, out_dim=3):

        super(TI_GenEmd_DS, self).__init__()

        self.input_dim = input_dim # noise dimension
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim # number of features map on the first layer
        self.out_dim = out_dim # number of channels

        self.noise_net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.input_dim, self.hidden_dim , 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            #nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim * 4),
            FReLU(in_channels=self.hidden_dim ),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.hidden_dim , self.hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim ),
            #nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim * 2),
            FReLU(in_channels=self.hidden_dim ),

            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            # nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim * 2),
            FReLU(in_channels=self.hidden_dim),

        )

        self.embed_net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.embedding_dim, self.hidden_dim *4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_dim *4),
            # nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim * 4),
            FReLU(in_channels=self.hidden_dim *4),

            nn.ConvTranspose2d(self.hidden_dim*4, self.hidden_dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim*2),
            # nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim * 2),
            FReLU(in_channels=self.hidden_dim*2),


            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.hidden_dim *2, self.hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim ),
            # nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim * 2),
            FReLU(in_channels=self.hidden_dim ),

        )
        self.se = SCALayer(channel=2 * self.hidden_dim, reduction=2)
        self.conv =  nn.Sequential(
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(2 * self.hidden_dim,  self.hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            # nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim ),
            FReLU(in_channels=self.hidden_dim),

            nn.ConvTranspose2d( self.hidden_dim, self.hidden_dim//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim//4),
            # nn.GroupNorm(num_groups=32, num_channels=self.hidden_dim ),
            FReLU(in_channels=self.hidden_dim//4),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.hidden_dim//4, self.out_dim, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.final = nn.Conv2d(out_dim,out_dim,1)

    def forward(self, noise, embedding):
        noise_re = self.noise_net(noise)
        #print('noise_re', noise_re.size())
        embed_re = self.embed_net(embedding)
        #print('embed_re', embed_re.size())

        r = torch.cat([noise_re, embed_re], dim=1)
        #print(r.size())
        se_ = self.se(r)
        se_ = self.conv(se_)
        ret = self.final(se_)
        return ret


if __name__ == '__main__':

    '''se =SELayer(channel=2, reduction=2)
    r1 = torch.randn(1, 1, 32, 32)
    r2 = torch.randn(1, 1, 32, 32)
    r = torch.cat([r1,r2],dim=1)
    print('r',r.size())
    rrrr = se(r)
    print('res', rrrr.size())

    exit()'''

    net = TI_GenEmd_DS(out_dim=1)
    print(sum(param.numel() for param in net.parameters()))
    tmpBatchSize = 1
    r = torch.randn(tmpBatchSize, 100, 1, 1)
    e = torch.randn(tmpBatchSize, 10, 1, 1)

    ret = net(r,e)
    print(ret.size())
