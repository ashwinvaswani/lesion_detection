import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from .functional import acs_conv_f
from collections import OrderedDict
from torch._six import container_abcs
import collections.abc
from itertools import repeat

# from .utiles import _to_triple, _triple_same, _pair_same

from .base_acsconv import _ACSConv

def as_triple(x, d_value=1): 
    if isinstance(x, container_abcs.Iterable):
        x = list(x)
        if len(x)==2:
            x = [d_value] + x
        return x
    else:
        return [d_value] + [x] * 2


def _ntuple_same(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            assert len(set(x))==1, 'the size of kernel must be the same for each side'
            return tuple(repeat(x[0], n))
    return parse

def _to_ntuple(n):
    def parse(x):
        if isinstance(x, int):
            return tuple(repeat(x, n))
        elif isinstance(x, collections.abc.Iterable):
            if len(set(x))==1:
                return tuple(repeat(x[0], n))
            else:
                assert len(x)==n , 'wrong format'
                return x
    return parse

_pair_same = _ntuple_same(2)
_triple_same = _ntuple_same(3)

_to_pair = _to_ntuple(2)
_to_triple = _to_ntuple(3)

class ACSConv(_ACSConv):
    """
    Vallina ACS Convolution
    
    Args:
        acs_kernel_split: optional, equally spit if not specified.

        Other arguments are the same as torch.nn.Conv3d.
    Examples:
        >>> import ACSConv
        >>> x = torch.rand(batch_size, 3, D, H, W)
        >>> conv = ACSConv(3, 10, kernel_size=3, padding=1)
        >>> out = conv(x)

        >>> conv = ACSConv(3, 10, acs_kernel_split=(4, 3, 3))
        >>> out = conv(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, acs_kernel_split=None, 
                 bias=True, padding_mode='zeros',n_fold=8):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, 0, groups, bias, padding_mode)
        if acs_kernel_split is None:
            if self.out_channels%3==0:
                self.acs_kernel_split = (self.out_channels//3,self.out_channels//3,self.out_channels//3)
            if self.out_channels%3==1:
                self.acs_kernel_split = (self.out_channels//3+1,self.out_channels//3,self.out_channels//3)
            if self.out_channels%3==2:
                self.acs_kernel_split = (self.out_channels//3+1,self.out_channels//3+1,self.out_channels//3)
        else:
            self.acs_kernel_split = acs_kernel_split

        # if int(self.out_channels / 3) > self.in_channels:
        #     self.pointwise = torch.nn.Conv3d(self.in_channels,self.out_channels,kernel_size=1)


    def forward(self, x):
        """
        Convolution forward function
        Divide the kernel into three parts on output channels based on acs_kernel_split, 
        and conduct convolution on three directions seperately. Bias is added at last.
        """
        # print("shape",self.weight.shape)
        return acs_conv_f(x, self.weight, self.bias, self.kernel_size, self.dilation, self.padding, self.stride, 
                            self.groups, self.out_channels, self.acs_kernel_split)
        # if int(self.out_channels / 3) > self.in_channels:
        #     print("in group")
        #     x = acs_conv_f(x, self.weight, self.bias, self.kernel_size, self.dilation, self.padding, self.stride, 
        #                         self.in_channels, self.in_channels, self.acs_kernel_split)
        #     x = self.pointwise(x)
        #     return x
        # else:
        #     return acs_conv_f(x, self.weight, self.bias, self.kernel_size, self.dilation, self.padding, self.stride, 
        #                     self.groups, self.out_channels, self.acs_kernel_split)



    def extra_repr(self):
        s = super().extra_repr() + ', acs_kernel_split={acs_kernel_split}'
        return s.format(**self.__dict__)
