import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import math
from math import inf, sqrt, pi
import numpy as np

def prod(x):
    p = 1
    for x_ in x:
        p *=x
    return p


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim,  bn=True, nonlinear='relu', dropout=0.,
            bias=True, init_type='standard', **kwargs):
        """A linear block.  The linear layer is followed by batch
        normalization (if active) and a ReLU (again, if active)

        Args:
            in_dim: number of input dimensions
            out_dim: number of output dimensions
            bn (bool, optional): turn on batch norm (default: False)
        """
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(th.randn(out_dim)) 
        else:
            self.register_parameter('bias', None)
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.nonlinear=nonlinear
        if bn:
            self.bn = nn.BatchNorm1d(out_dim, affine=False)
        else:
            self.bn = False

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.init_type = init_type
        if self.init_type == 'standard':
            self.reset_parameters()
        elif self.init_type == 'xavier':
            nn.init.xavier_normal_(self.weight.data)
        elif self.init_type == 'kaiming':
            nn.init.kaiming_normal(self.weight.data,mode='fan_in',nonlinearity='relu')
        elif self.init_type == 'zero_init':
            self.weight.data = nn.Parameter(th.zeros(out_dim,in_dim))

    def reset_parameters(self):
        n = self.in_dim
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)

        y = F.linear(x, self.weight, None)


        if self.bias is not None:
            b = self.bias.view(1,-1)
            y = y+b

        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        if self.bn:
            y = self.bn(y)

        return y

    def extra_repr(self):
        s = ('{in_dim}, {out_dim}')

        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,  stride=1, padding=None,
            kernel_size=(3,3),  bn=True, nonlinear='relu', dropout = 0.,
            init_type = 'standard', bias=True,**kwargs):
        """A 2d convolution block.  The convolution is followed by batch
        normalization (if active).

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride (int, optional): stride of the convolutions (default: 1)
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)
        """
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(th.randn(out_channels))
        else:
            self.register_buffer('bias', None)
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size=_pair(kernel_size)
        self.nonlinear=nonlinear
        self.num_layers=num_layers
        if padding is None:
            self.padding = tuple([k//2 for k in kernel_size])
        else:
            self.padding = _pair(padding)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels, affine=False)
        else:
            self.bn = False

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.init_type = init_type
        if self.init_type == 'standard':
            self.reset_parameters()
        elif self.init_type == 'xavier':
            nn.init.xavier_normal_(self.weight.data)
        elif self.init_type == 'kaiming':
            nn.init.kaiming_normal(self.weight.data,mode='fan_in',nonlinearity='relu')
        elif self.init_type == 'zero_init':
            self.weight.data = nn.Parameter(th.zeros(out_channels,in_channels,*kernel_size))

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)

        y = F.conv2d(x, self.weight, None, self.stride, self.padding,
                1, 1)

        if self.bias is not None:
            b = self.bias.view(1,self.out_channels,1,1)
            y = y+b

        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        if self.bn:
            y = self.bn(y)

        return y

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'
        return s.format(**self.__dict__)



class BasicBlock(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), bn=True, nonlinear='relu',
            dropout=0., residual=True, weight_init='standard', zero_last=False, **kwargs):
        """A basic 2d ResNet block, with modifications on original ResNet paper
        [1].  Every convolution is followed by batch normalization (if active).

        Args:
            channels: number of input and output channels
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)

        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016.
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = _pair(kernel_size)
        self.nonlinear='relu'
        self.residual = residual

        if bn:
            self.bn = nn.BatchNorm2d(channels, affine=False)
        else:
            self.bn = False

        self.conv0 = Conv(channels, channels, kernel_size=kernel_size, bn=bn, dropout=dropout,
                nonlinear=nonlinear, init_type = weight_init)
        if zero_last:
            self.conv1 = Conv(channels, channels, kernel_size=kernel_size, bn=False, dropout=dropout,
                    nonlinear=False,init_type='zero_init')
        else:
            self.conv1 = Conv(channels, channels, kernel_size=kernel_size, bn=False, dropout=dropout,
                    nonlinear=False,init_type=weight_init)

    def forward(self, x):

        y = self.conv0(x)
        y = self.conv1(y)

        if self.residual:
            y = x+y

        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        if self.bn:
            y = self.bn(y)

        return y

    def extra_repr(self):
        s = ('{channels}, {channels}, kernel_size={kernel_size}')
        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

class Bottleneck(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), bn=True, nonlinear='relu',
            dropout=0., residual=True, weight_init='standard', zero_last=False,**kwargs):
        """A basic 2d ResNet bottleneck block, with modifications on original ResNet paper
        [1].  Every convolution is followed by batch normalization (if active).

        Args:
            channels: number of input and output channels
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)

        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016.
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = _pair(kernel_size)
        self.nonlinear=nonlinear
        self.residual = residual
        if bn:
            self.bn = nn.BatchNorm2d(channels, affine=False)
        else:
            self.bn = False


        self.conv0 = Conv(channels, channels//4, kernel_size=(1,1), bn=bn, dropout=dropout,
                nonlinear=nonlinear,init_type=weight_init)
        self.conv1 = Conv(channels//4, channels//4, kernel_size=kernel_size, bn=bn, dropout=dropout,
                nonlinear=nonlinear,init_type=weight_init)
        if zero_last:
            self.conv2 = Conv(channels//4, channels, kernel_size=(1,1), bn=False, dropout=dropout,
                    nonlinear=False,init_type='zero_init')
        else:
            self.conv2 = Conv(channels//4, channels, kernel_size=(1,1), bn=False, dropout=dropout,
                    nonlinear=False,init_type=weight_init)


    def forward(self, x):

        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)

        if self.residual:
            y = x+y

        if self.nonlinear=='smoothrelu':
            y = smoothrelu(y)
        elif self.nonlinear=='softplus':
            y = F.softplus(y)
        elif self.nonlinear=='leaky_relu':
            y = F.leaky_relu(y)
        elif self.nonlinear=='selu':
            y = F.selu(y)
        elif self.nonlinear=='elu':
            y = F.elu(y)
        elif self.nonlinear:
            y = F.relu(y)

        if self.bn:
            y = self.bn(y)

        return y


    def extra_repr(self):
        s = ('{channels}, {channels}, kernel_size={kernel_size}')
        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

