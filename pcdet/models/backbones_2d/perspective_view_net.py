import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_utils.weight_init import *
from ..model_utils.layers import ConvBNReLU

BN_EPS = 1e-3
BN_MOMENTUM = 0.01


def upsample(idims, odims, stride=(2 ,2), mode='deconv'):
  if mode == 'deconv':
    return  nn.Sequential(
              nn.ConvTranspose2d(
                idims, odims,
                stride,
                stride=stride, bias=False
              ),
              nn.BatchNorm2d(odims, eps=BN_EPS, momentum=BN_MOMENTUM)
            )
    
  else:
    return nn.Sequential(
            nn.Conv2d(idims, odims, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(odims, eps=BN_EPS, momentum=BN_MOMENTUM),
            nn.Upsample(scale_factor=stride, mode='nearest')
           )


class iBottleneckBlock(nn.Module):
  """Inverted Bottleneck Block."""

  def __init__(self, idims, odims, stride=1):
    super(iBottleneckBlock, self).__init__()
    
    self.conv1 = nn.Sequential(
      nn.Conv2d(idims, idims, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(idims, eps=BN_EPS, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )
    
    if idims == odims and stride == 1:
      self.conv2 = nn.Sequential(
        nn.Conv2d(idims, 6*idims, 3, stride=1, padding=1, groups=idims, bias=False),
        nn.BatchNorm2d(6*idims, eps=BN_EPS, momentum=BN_MOMENTUM),
        nn.Conv2d(6*idims, odims, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(odims, eps=BN_EPS, momentum=BN_MOMENTUM)
      )  
      self.shortcut = None
    else:
      self.conv2 = nn.Sequential(
        nn.Conv2d(idims, 6*idims, 3, stride=2, padding=1, groups=idims, bias=False),
        nn.BatchNorm2d(6*idims, eps=BN_EPS, momentum=BN_MOMENTUM),
        nn.Conv2d(6*idims, 6*idims, 3, stride=1, padding=1, groups=6*idims, bias=False),
        nn.BatchNorm2d(6*idims, eps=BN_EPS, momentum=BN_MOMENTUM),
        nn.Conv2d(6*idims, odims, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(odims, eps=BN_EPS, momentum=BN_MOMENTUM)
      )
      self.shortcut = nn.Sequential(
        nn.Conv2d(idims, 6*idims, 3, stride=2, padding=1, groups=idims, bias=False),
        nn.BatchNorm2d(6*idims, eps=BN_EPS, momentum=BN_MOMENTUM),
        nn.Conv2d(6*idims, odims, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(odims, eps=BN_EPS, momentum=BN_MOMENTUM)
      )

    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    if self.shortcut is not None:
      shortcut = self.shortcut(x)
    else:
      shortcut = x

    x = self.conv1(x)
    x = self.conv2(x)
    x = x + shortcut
    x = self.relu(x)
    return x


class iRangeViewNet(nn.Module):
  """
  RangeViewNet with Inverted Bottleneck Block.
  """

  def __init__(self, in_channels, init_type='kaiming_uniform'):
    super(iRangeViewNet, self).__init__()

    self.res1 = iBottleneckBlock(idims=in_channels, odims=in_channels)
    self.res2 = iBottleneckBlock(idims=in_channels, odims=2*in_channels, stride=(2, 2))
    self.res3 = iBottleneckBlock(idims=2*in_channels, odims=4*in_channels, stride=(2, 2))
    self.deconv2 = upsample(2*in_channels, odims=in_channels, stride=(2, 2))
    self.deconv3 = upsample(4*in_channels, odims=in_channels, stride=(4, 4))
    self.conv_out = nn.Conv2d(3*in_channels, in_channels, 3, stride=1, padding=1)
    self.init_weights(init_type)

  def init_weights(self, init_type):
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if init_type == 'kaiming_uniform':
          kaiming_init(m, distribution='uniform')
        elif init_type == 'kaiming_normal':
          kaiming_init(m, distribution='normal')
        elif init_type == 'xavier':
          xavier_init(m)
        elif init_type =='caffe2_xavier':
          caffe2_xavier_init(m)
          
      elif isinstance(m, nn.BatchNorm2d):
        constant_init(m, 1)

  def forward(self, voxels_in):
    voxels_out1 = self.res1(voxels_in)
    voxels_out2 = self.res2(voxels_in)
    voxels_out3 = self.res3(voxels_out2)
    
    voxels_out2 = self.deconv2(voxels_out2)
    voxels_out3 = self.deconv3(voxels_out3)
    
    voxels_out = torch.cat([voxels_out1, voxels_out2, voxels_out3], axis=1)
    voxels_out = self.conv_out(voxels_out)

    return voxels_out
    
