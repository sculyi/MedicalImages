'''
author: lyi
date 20210129


desc classififer model
TODO revise the module name
'''

# torch imports
import torch
from torch import nn
from conv_SAME_pad import Conv2d_samepadding

class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.depthwise_conv_bn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels))

    def forward(self, x):
        funnel_x = self.depthwise_conv_bn(x)
        return torch.max(x, funnel_x)

class MSNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, merge_mode='concat', stride = 2):
        super(MSNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim =hidden_dim
        self.merge_mode = 'concat' if merge_mode is None else merge_mode
        self.stride = stride
        self.make_conv()


    def make_conv(self):
        self.conv_s1 = nn.Sequential(
            Conv2d_samepadding(in_channels=self.input_dim,out_channels=self.hidden_dim, kernel_size = (1,1), stride=self.stride,bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            FReLU(self.hidden_dim))
        self.conv_s2 = nn.Sequential(
            Conv2d_samepadding(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=(3, 3),
                                          stride=self.stride,bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            FReLU(self.hidden_dim))
        self.conv_s3 = nn.Sequential(
            Conv2d_samepadding(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=(5, 5),
                                          stride=self.stride,bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            FReLU(self.hidden_dim))

    def forward(self, x):
        ms_conv1 = self.conv_s1(x)
        ms_conv2 = self.conv_s2(x)
        ms_conv3 = self.conv_s3(x)
        if self.merge_mode == 'concat':
            final = torch.cat([ms_conv1, ms_conv2, ms_conv3], dim=1)
        elif self.merge_mode == 'mean':
            final = torch.mean(torch.stack([ms_conv1, ms_conv2, ms_conv3]), 0)

        return final


class SCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCALayer, self).__init__()
        reduction = min(channel,reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.SA = nn.Sequential(nn.AdaptiveAvgPool2d((None, None)),
                                Conv2d_samepadding(channel, channel//2, (3, 3), stride=1, bias=False),
                                nn.BatchNorm2d(num_features=channel//2),
                                nn.PReLU(channel//2),
                                Conv2d_samepadding(channel//2, 1, (3, 3), stride=1, bias=False),
                                nn.BatchNorm2d(num_features=1),
                                nn.Sigmoid())

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        sa_ = self.SA(x)
        #TODO for test only, save the learned weights
        #save_to_h5(sa_)

        return x*y*sa_





class MSCANet(nn.Module):
    def __init__(self, input_dim, hidden_dim, reduction, merge_mode=None):
        super(MSCANet, self).__init__()
        self.pre = nn.Sequential(Conv2d_samepadding(input_dim, hidden_dim,(3,3),stride=2,bias=False),
                                 nn.BatchNorm2d(hidden_dim),
                                 FReLU(hidden_dim)) #64

        self.block1 = nn.Sequential(MSNet(hidden_dim, hidden_dim, merge_mode), #64
                                   SCALayer(3 * hidden_dim, reduction),
	                                nn.BatchNorm2d(3*hidden_dim),
                                    Conv2d_samepadding(3 * hidden_dim, hidden_dim, (1, 1), stride=2,bias=False) )#32

        self.block1 = nn.Sequential(Conv2d_samepadding(hidden_dim, 2*hidden_dim, (3,3),stride=2,bias=False),  # 32
                                    nn.BatchNorm2d( 2*hidden_dim),
                                    FReLU(2*hidden_dim),
                                    SCALayer(2*hidden_dim, reduction,size=32),
                                    nn.BatchNorm2d( 2*hidden_dim),
                                    Conv2d_samepadding(2*hidden_dim, 4*hidden_dim, (3, 3), stride=2, bias=False),  # 16
                                    nn.BatchNorm2d(4*hidden_dim),
                                    FReLU(4 * hidden_dim),
                                    SCALayer(4*hidden_dim, reduction,size=16),
                                    nn.BatchNorm2d(4*hidden_dim),
                                    Conv2d_samepadding(4 * hidden_dim, 4 * hidden_dim, (3, 3), stride=2, bias=False),  #8
                                    nn.BatchNorm2d(4 * hidden_dim),
                                    FReLU(4 * hidden_dim),
                                    SCALayer(4 * hidden_dim, reduction,size=8),
                                    nn.BatchNorm2d(4 * hidden_dim),
                                    Conv2d_samepadding(4 * hidden_dim, 2 * hidden_dim, (3, 3), stride=2, bias=False),  #4
                                    nn.BatchNorm2d(2 * hidden_dim),
                                    FReLU(2 * hidden_dim),

                                    )


        third_ = hidden_dim//3
        self.block2 = nn.Sequential(MSNet(hidden_dim, third_, merge_mode,stride=1),
                                    SCALayer(3*third_, reduction//3),
                                   Conv2d_samepadding(3*third_, 1, (1, 1), stride=1,bias=False),
                                    nn.BatchNorm2d(16))

        self.block2 = nn.Sequential(MSNet(hidden_dim, hidden_dim, merge_mode),#16
                                   SCALayer(3 * hidden_dim, reduction),
	                                nn.BatchNorm2d(3*hidden_dim),
                                    Conv2d_samepadding(3 * hidden_dim, 2*hidden_dim, (1, 1), stride=2,bias=False),#8
                                    nn.BatchNorm2d(2*hidden_dim),
                                    Conv2d_samepadding(2*hidden_dim, 2*hidden_dim,(1,1),stride=2,bias=False),#4
                                    nn.BatchNorm2d(2*hidden_dim),)
        pass


    def forward(self, x):
        x = self.pre(x)
        x_ = self.block1(x)
        #print(x_.size())
        #x_ = self.block2(x_)
        #print(x_.size())
        #x_ = self.block3(x_)
        return x_


import numpy as np
if __name__ == '__main__':
    test = np.array([1,2,3]).reshape((1,3,1,1))
    test = torch.from_numpy(test)

    yyy = np.array([[1,2],[3,4]]).reshape(1,1,2,2)
    yyy = torch.from_numpy(yyy)
    tmp = test*yyy

    b,c,w,h=1,3,128,128
    hidden = 32
    input_data = torch.rand((b,c,w,h))
    aaa = test.expand_as(input_data)
    aap =  nn.Sequential(nn.Conv2d(c,1,(1,1),bias=False),
            FReLU(1),
            nn.AdaptiveAvgPool2d(w),
            nn.Conv2d(1, 1, (1, 1), bias=False))
    print(aap(input_data).size())
    exit()

    net = MSCANet(c,hidden,4)#SCALayer(c,1)#
    out = net(input_data)
    print('out', out.size())
    pass