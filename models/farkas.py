import math
import numpy as np

import torch as th
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from .utils import View, Avg2d
from .blocks import Conv

class FarkasLinear(nn.Module):
    def __init__(self, in_dim, out_dim,  bn=True, nonlinear=True, dropout=0.,
                init_type='standard',**kwargs):
        """A linear block, with guaranteed non-zero gradient.  The linear layer
        is followed by batch normalization (if active) and a ReLU (again, if
        active)

        Args:
            in_dim: number of input dimensions
            out_dim: number of output dimensions
            bn (bool, optional): turn on batch norm (default: False)
        """
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_dim-1, in_dim))
        self.bias = nn.Parameter(th.randn(out_dim))
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
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)

        y = F.linear(x, self.weight, None)

        ybar = (-y).mean(dim=1,keepdim=True)

        y = th.cat([y,ybar],dim=1)

        bbar = th.max(-(self.bias[0:-1]).mean(),self.bias[-1])
        b = th.cat([self.bias[0:-1],bbar.unsqueeze(0)],dim=0)

        y = y + b.view(1,self.out_dim)

        if self.nonlinear=='leaky_relu':
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



class FarkasConv(nn.Module):
    def __init__(self, in_channels, out_channels,  stride=1, padding=None,
            kernel_size=(3,3),  bn=True, nonlinear=True, dropout=0., 
            init_type='standard',**kwargs):
        """A 2d convolution block, with guaranteed non-zero gradient.  The
        convolution is followed by batch normalization (if active).

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            stride (int, optional): stride of the convolutions (default: 1)
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)
        """
        super().__init__()

        if out_channels <2:
            raise ValueError('need out_channels>=2')

        self.weight = nn.Parameter(th.randn(out_channels-1, in_channels, *kernel_size))
        self.bias = nn.Parameter(th.randn(out_channels))
        self.stride = stride
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size=_pair(kernel_size)
        if padding is None:
            self.padding = tuple([k//2 for k in kernel_size])
        else:
            self.padding = _pair(padding)
        self.nonlinear = nonlinear

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
            self.weight.data = nn.Parameter(th.zeros(out_channels-1, in_channels, *kernel_size))

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):

        if self.dropout:
            x = self.dropout(x)

        y = F.conv2d(x, self.weight, None, self.stride, self.padding,
                1, 1)

        ybar = (-y).mean(dim=1,keepdim=True)
        y = th.cat([y,ybar],dim=1)
        bbar = th.max( - (self.bias[0:-1]).mean() , self.bias[-1])
        b = th.cat([self.bias[0:-1],bbar.unsqueeze(0)],dim=0)

        y = y + b.view(1,self.out_channels,1,1)

        if self.nonlinear=='leaky_relu':
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


class FarkasBlock(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), bn=True, nonlinear=True,
            dropout = 0.,  residual=True, weight_init='standard',zero_last=False,**kwargs):
        """A basic 2d ResNet block, with modifications on original ResNet paper
        [1].  Every convolution is followed by batch normalization (if active).
        The gradient is guaranteed to be non-zero.

        Args:
            channels: number of input and output channels
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)

        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016.
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
        """
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels+1
        self.kernel_size = _pair(kernel_size)
        self.nonlinear=nonlinear
        self.residual = residual
        self.conv0 = FarkasConv(channels, channels,
                kernel_size=kernel_size, bn=bn, nonlinear=nonlinear, init_type=weight_init)

        if zero_last:
            self.weight = nn.Parameter(th.zeros(channels,channels,*kernel_size))
            self.bias=nn.Parameter(th.zeros(channels+1))
        else:
            self.weight = nn.Parameter(th.randn(channels, channels, *kernel_size))
            self.bias = nn.Parameter(th.randn(channels+1))
        self.padding = tuple([k//2 for k in kernel_size])
        if bn:
            self.bn = nn.BatchNorm2d(channels+1, affine=False)
        else:
            self.bn = False

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.init_type = weight_init
        if not zero_last:
            if self.init_type == 'standard':
                self.reset_parameters()
            elif self.init_type == 'xavier':
                nn.init.xavier_normal_(self.weight.data)
            elif self.init_type == 'kaiming':
                nn.init.kaiming_normal(self.weight.data,mode='fan_in',nonlinearity='relu')


    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)



    def forward(self, x):

        if self.dropout:
            y = self.dropout(x)
        else:
            y=x

        y = self.conv0(y)

        if self.dropout:
            y = self.dropout(y)

        y = F.conv2d(y, self.weight, None, 1, self.padding,
                1, 1)

        if self.residual:
            ybar = (-x-y).mean(dim=1,keepdim=True)
            y = th.cat([x+y,ybar],dim=1)
        else:
            ybar = (-y).mean(dim=1,keepdim=True)
            y = th.cat([y,ybar],dim=1)

        bbar = th.max( - (self.bias[0:-1]).mean(),self.bias[-1])
        b = th.cat([self.bias[0:-1],bbar.unsqueeze(0)],dim=0)

        y = y + b.view(1,self.out_channels,1,1)

        if self.nonlinear=='leaky_relu':
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
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}')

        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

class FarkasBottleneck(nn.Module):

    def __init__(self, channels, kernel_size=(3,3), bn=True, nonlinear=True,
            dropout = 0., residual=True, weight_init='standard',zero_last=False,**kwargs):
        """A basic 2d ResNet block, with modifications on original ResNet paper
        [1].  Every convolution is followed by batch normalization (if active).
        The gradient is guaranteed to be non-zero.

        Args:
            channels: number of input and output channels
            kernel_size (tuple, optional): kernel shape (default: 3)
            bn (bool, optional): turn on batch norm (default: False)

        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, 2016.
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
        """
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels+1
        self.kernel_size = _pair(kernel_size)
        self.nonlinear = nonlinear
        self.residual = residual

        self.conv0 = FarkasConv(channels, channels//4,
                kernel_size=(1,1), bn=bn,
                nonlinear=nonlinear, init_type=weight_init)
        self.conv1 = FarkasConv(channels//4, channels//4,
                kernel_size=kernel_size, bn=bn,
                nonlinear=nonlinear,init_type=weight_init)

        if zero_last:
            self.weight = nn.Parameter(th.zeros(channels,channels//4, 1,1))
            self.bias = nn.Parameter(th.zeros(channels+1))
        else:
            self.weight = nn.Parameter(th.randn(channels, channels//4, 1,1))
            self.bias = nn.Parameter(th.randn(channels+1))
        if bn:
            self.bn = nn.BatchNorm2d(channels+1, affine=False)
        else:
            self.bn = False

        if dropout>0.:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.init_type = weight_init
        if self.init_type == 'standard':
            self.reset_parameters()
        elif self.init_type == 'xavier':
            nn.init.xavier_normal_(self.weight.data)
        elif self.init_type == 'kaiming':
            nn.init.kaiming_normal(self.weight.data,mode='fan_in',nonlinearity='relu')

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):

        if self.dropout:
            y = self.dropout(x)
        else:
            y=x

        y = self.conv0(y)

        if self.dropout:
            y = self.dropout(y)

        y = self.conv1(y)

        if self.dropout:
            y = self.dropout(y)

        y = F.conv2d(y, self.weight, None, 1, 0,
                1, 1)

        if self.residual:
            ybar = (-x - y).mean(dim=1,keepdim=True)
            y = th.cat([x+y,ybar],dim=1)
        else:
            ybar = (-y).mean(dim=1,keepdim=True)
            y = th.cat([y,ybar],dim=1)

        bbar = th.max(-(self.bias[0:-1]).mean(),self.bias[-1])
        b = th.cat([self.bias[0:-1],bbar.unsqueeze(0)],dim=0)

        y = y + b.view(1,self.out_channels,1,1)

        if self.nonlinear=='leaky_relu':
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
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}')

        if self.bn:
            s += ', batchnorm=True'
        else:
            s += ', batchnorm=False'

        return s.format(**self.__dict__)

class FarkasNet(nn.Module):

    def __init__(self, layers, block=FarkasBlock, in_channels=3,
                 classes=10, kernel_size=(3,3), nonlinear=True,
                 conv0_kwargs = {'kernel_size':(3,3), 'stride':1},
                 conv0_pool=None, downsample_pool=nn.AvgPool2d,
                 last_layer_nonlinear=False, last_layer_bn=None,
                 dropout=0.,weight_init='standard',zero_last=False,
                 bn=True, base_channels=16, **kwargs):
        if last_layer_bn is None:
            last_layer_bn=bn

        super().__init__()
        kernel_size = _pair(kernel_size)

        def make_layer(n, block, in_channels, out_channels, stride):
            sublayers = []
            if not in_channels==out_channels:
                conv = FarkasConv
                sublayers.append(conv(in_channels, out_channels, kernel_size=(1,1),
                    nonlinear=True, dropout=dropout, bn=bn,init_type=weight_init))

            if stride>1:
                sublayers.append(downsample_pool(stride))

            for k in range(n):
                u = k
                sublayers.append(block(out_channels+u, kernel_size=kernel_size, dropout=dropout,
                    bn=bn, nonlinear=nonlinear, weight_init=weight_init,zero_last=zero_last,**kwargs))

            return nn.Sequential(*sublayers)


        conv = FarkasConv
        pdsz = [k//2 for k in conv0_kwargs['kernel_size'] ]
        self.layer0 = conv(in_channels, base_channels, padding=pdsz,
                **conv0_kwargs, dropout=dropout, bn=bn, nonlinear=nonlinear,weight_init=weight_init)

        if conv0_pool:
            self.maxpool = conv0_pool
        else:
            self.maxpool = False


        _layers = []
        for i, n in enumerate(layers):

            if i==0:
                _layers.append(make_layer(n, block, base_channels,
                    base_channels, 1))
            else:
                u = layers[i-1]
                _layers.append(make_layer(n, block, base_channels*(2**(i-1))+u,
                    base_channels*(2**i), 2))

        self.layers = nn.Sequential(*_layers)

        self.pool = Avg2d()
        u = layers[-1]
        self.view = View((2**i)*base_channels+u)

        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = False

        self.fc = nn.Linear((2**i)*base_channels+u,classes)
        
        self.nonlinear=nonlinear
        self.bn = bn

    @property
    def num_parameters(self):
        return sum([w.numel() for w in self.parameters()])


    def forward(self, x):
        x = self.layer0(x)
        if self.maxpool:
            x = self.maxpool(x)
        x = self.layers(x)
        x = self.pool(x)
        x = self.view(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x

def FarkasNet18(**kwargs):
    m = FarkasNet([3,3,3],block=FarkasBlock,**kwargs)
    return m

def FarkasNet50(**kwargs):
    m = FarkasNet([3,4,6,3],base_channels=64,block=FarkasBottleneck,**kwargs)
    return m

def FarkasNet101(**kwargs):
    m = FarkasNet([3,4,23,3],base_channels=64,block=FarkasBottleneck,**kwargs)
    return m

def FarkasNet110(**kwargs):
    m = FarkasNet([18,18,18],block=FarkasBlock,**kwargs)

def FarkasNet34(**kwargs):
    m = FarkasNet([5,5,5],block=FarkasBlock,**kwargs)
    return m
