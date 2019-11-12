from torch import nn
from .resnet import ResNet
from .blocks import Conv

def ResNet18(**kwargs):
    m = ResNet([2,2,2,2],**kwargs)
    return m

def ResNet34(**kwargs):
    m = ResNet([5,5,5],**kwargs)
    return m

def ResNet110(**kwargs):
    m = ResNet([18,18,18],**kwargs)
    return m

def ResNet50(**kwargs):
    m = ResNet([3,4,6,3],base_channels=64, block='Bottleneck',**kwargs)
    return m

def ResNet101(**kwargs):
    m = ResNet([3,4,23,3],base_channels=64, block='Bottleneck',**kwargs)
    return m
