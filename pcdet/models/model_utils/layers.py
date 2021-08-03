import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, eps=1e-3, momentum=0.01):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False),
      nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
      nn.ReLU(inplace=True)
    )
  
  def forward(self, x):
    out = self.block(x)
    return out


class SeparateConvBNReLU(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, eps=1e-3, momentum=0.01):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, \
                        padding=kernel_size//2, groups=in_channels, bias=False),
      nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
      nn.ReLU(inplace=True)
    )
  
  def forward(self, x):
    out = self.block(x)
    return out