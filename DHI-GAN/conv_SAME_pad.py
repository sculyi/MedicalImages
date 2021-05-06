'''
author: lyi
date 20210129
desc basic lock for the conv (same padding) in torch

'''

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter



def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows, input_cols = input.size(2), input.size(3)
    filter_rows, filter_cols = weight.size(2), weight.size(3)


    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)

    out_cols = (input_cols + stride[1]-1 ) // stride[1]

    padding_cols = max(0, (out_cols - 1) * stride[1] +
                        (filter_cols - 1) * dilation[1] + 1 - input_cols)
    cols_odd = (padding_cols % 2 != 0)

    if rows_odd or cols_odd:
        input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


def conv1d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)


    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)

    #print('Conv1D_SAME', input_rows, filter_rows, out_rows, padding_rows, rows_odd)

    if rows_odd:
        input = F.pad(input, [0, int(rows_odd), 0, 0])


    return F.conv1d(input, weight, bias, stride,
                  padding=padding_rows // 2,
                  dilation=dilation, groups=groups)

class _ConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, device):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))#.to(device)
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))#.to(device)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))#.to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)



class Conv2d_samepadding(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, device=''):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d_samepadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, device)
    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv1d_samepadding(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, device=''):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d_samepadding, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, device)
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



if __name__ == '__main__':
    import numpy as np
    b, t,f = 2, 117,39
    a = np.random.random((b,1,t,f ))
    a = torch.FloatTensor(a)

    k = 21

    b = Conv2d_samepadding(1,32, (k,k),(2,2))(a)
    c = nn.Conv2d(1,32, (k,k),(2,2))(a)
    print(b.size(), c.size())


    a1 = torch.squeeze(a).permute((0,2,1))
    b1 = Conv1d_samepadding(f,64, kernel_size=k, stride=2)(a1)
    b2 = nn.Conv1d(f,64, kernel_size=k, stride=2)(a1)
    print(a1.size(),b1.size(), b2.size())

