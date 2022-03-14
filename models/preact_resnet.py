'''
Original PreActResNet Implementation from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
extended and modified to support stochastic depth, final layer dropout, squeeze-excitation and shakedrop
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from base_model import BaseModel
from timm.models.layers import DropPath, create_attn
from regularization.shakedrop import ShakeDrop


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, stochastic_depth=0.0, apply_se=False, p_shakedrop=None):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # Stochastic Depth
        self.drop_path = DropPath(drop_prob=stochastic_depth)
        # Squeeze and Excitation
        self.apply_se = apply_se
        outplanes = planes * self.expansion
        if self.apply_se:
            self.se = create_attn('se', outplanes, reduction_ratio=0.25) # ratio from https://arxiv.org/pdf/2103.07579.pdf
        # Shakedrop
        self.shake_drop = ShakeDrop(p_shakedrop) if p_shakedrop else None


        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        if self.apply_se:
            out = self.se(out)
        out = self.drop_path(out)
        if self.shake_drop:
            out = self.shake_drop(out)

        out = out + shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, stochastic_depth=0.0, apply_se=False, p_shakedrop=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        # Stochastic Depth
        self.drop_path = DropPath(drop_prob=stochastic_depth)
        # Squeeze and Excitation
        self.apply_se = apply_se
        outplanes = planes * self.expansion
        if self.apply_se:
            self.se = create_attn('se', outplanes, reduction_ratio=0.25)  # ratio from https://arxiv.org/pdf/2103.07579.pdf
        # Shakedrop
        self.shake_drop = ShakeDrop(p_shakedrop) if p_shakedrop else None

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))

        if self.apply_se:
            out = self.se(out)
        out = self.drop_path(out)
        if self.shake_drop:
            out = self.shake_drop(out)

        out = out + shortcut
        return out


class PreActResNet(BaseModel):
    def __init__(self, block, num_blocks, num_classes=10, hypparams={}):
        super(PreActResNet, self).__init__(hypparams)
        self.in_planes = 64

        # shakedrop
        n = int(np.sum(num_blocks))
        self.u_idx = 0
        if self.apply_shakedrop:
            self.ps_shakedrop = [1 - (1.0 - (0.5 / n) * (i + 1)) for i in range(n)]
        else:
            self.ps_shakedrop = [None for _ in range(n)]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.stochastic_depth, self.se, self.ps_shakedrop[self.u_idx]))
            self.u_idx += 1
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)

        out = F.dropout(out, p=self.resnet_dropout, training=self.training)

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def PreActResNet18(num_classes, hypparams):
    if not hypparams['bottleneck']:
        return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, hypparams=hypparams)


def PreActResNet34(num_classes, hypparams):
    if not hypparams['bottleneck']:
        return PreActResNet(PreActBlock, [3,4,6,3], num_classes=num_classes, hypparams=hypparams)


def PreActResNet50(num_classes, hypparams):
    if hypparams['bottleneck']:
        return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes=num_classes, hypparams=hypparams)
    else:
        return PreActResNet(PreActBlock, [4, 6, 10, 5], num_classes=num_classes, hypparams=hypparams)


def PreActResNet101(num_classes, hypparams):
    if hypparams['bottleneck']:
        return PreActResNet(PreActBottleneck, [3,4,23,3], num_classes=num_classes, hypparams=hypparams)


def PreActResNet152(num_classes, hypparams):
    if hypparams['bottleneck']:
        return PreActResNet(PreActBottleneck, [3,8,36,3], num_classes=num_classes, hypparams=hypparams)
    else:
        return PreActResNet(PreActBlock, [4, 13, 55, 4], num_classes=num_classes, hypparams=hypparams)
