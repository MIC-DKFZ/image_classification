"""
Wide ResNet adapted from https://github.com/mzhaoshuai/Divide-and-Co-training/blob/main/model/resnet.py
extended and modified to support stochastic depth, final layer dropout, squeeze-excitation and shakedrop
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, create_attn

from base_model import BaseModel
from regularization.shakedrop import ShakeDrop


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        base_width=64,
        dilation=1,
        norm_layer=None,
        stochastic_depth=0.0,
        apply_se=False,
        p_shakedrop=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if base_width != 64:
            raise ValueError("BasicBlock only supports base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        # Stochastic Depth
        self.drop_path = DropPath(drop_prob=stochastic_depth)
        # Squeeze and Excitation
        self.apply_se = apply_se
        outplanes = planes * self.expansion
        if self.apply_se:
            self.se = create_attn(
                "se", outplanes, reduction_ratio=0.25
            )  # ratio from https://arxiv.org/pdf/2103.07579.pdf
        # Shakedrop
        self.shake_drop = ShakeDrop(p_shakedrop) if p_shakedrop else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.apply_se:
            out = self.se(out)
        out = self.drop_path(out)
        if self.shake_drop:
            out = self.shake_drop(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        base_width=64,
        dilation=1,
        norm_layer=None,
        stochastic_depth=0.0,
        apply_se=False,
        p_shakedrop=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0))

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # Stochastic Depth
        self.drop_path = DropPath(drop_prob=stochastic_depth)
        # Squeeze and Excitation
        self.apply_se = apply_se
        outplanes = planes * self.expansion
        if self.apply_se:
            self.se = create_attn(
                "se", outplanes, reduction_ratio=0.25
            )  # ratio from https://arxiv.org/pdf/2103.07579.pdf
        # Shakedrop
        self.shake_drop = ShakeDrop(p_shakedrop) if p_shakedrop else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.apply_se:
            out = self.se(out)
        out = self.drop_path(out)
        if self.shake_drop:
            out = self.shake_drop(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(BaseModel):
    def __init__(
        self,
        arch,
        block,
        layers,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        output_stride=8,
        num_classes=10,
        hypparams={},
    ):
        super(ResNet, self).__init__(**hypparams)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.cifar_size = hypparams["cifar_size"]
        # Shakedrop
        if self.cifar_size:
            n = int(np.sum(layers[:-1]))
        else:
            n = int(np.sum(layers))
        self.u_idx = 0
        if self.apply_shakedrop:
            self.ps_shakedrop = [1 - (1.0 - (0.5 / n) * (i + 1)) for i in range(n)]
        else:
            self.ps_shakedrop = [None for _ in range(n)]

        self.inplanes = 16
        self.base_width = width_per_group
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )

        if not self.cifar_size:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    self.inplanes,
                    self.inplanes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
                # output channle = inplanes * 2
                nn.Conv2d(
                    self.inplanes,
                    self.inplanes * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                norm_layer(self.inplanes * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            inplanes_origin = self.inplanes
            # 64 -> 128
            self.inplanes = self.inplanes * 2
            strides = [1, 2, 2, 2]
        # n_channels = [64, 128, 256, 512]

        else:
            # for training cifar, change the kernel_size=7 -> kernel_size=3 with stride=1
            self.layer0 = nn.Sequential(
                # nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(self.inplanes),
                nn.ReLU(inplace=True),
            )
            inplanes_origin = self.inplanes

            widen_factor = float(arch.split("_")[-1])
            inplanes_origin = inplanes_origin * int(max(widen_factor / (1**0.5) + 0.4, 1.0))

            # 32 -> 32 -> 16 -> 8
            strides = [1, 2, 2, 1]
            if output_stride == 2:
                print("INFO:PyTorch: Using output_stride {} on cifar10".format(output_stride))
                strides = [1, 1, 2, 1]

        self.layer1 = self._make_layer(block, inplanes_origin, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(
            block,
            inplanes_origin * 2,
            layers[1],
            stride=strides[1],
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            inplanes_origin * 4,
            layers[2],
            stride=strides[2],
            dilate=replace_stride_with_dilation[1],
        )
        inplanes_now = inplanes_origin * 4

        # If dataset is cifar, do not use layer4 because the size of the feature map is too small.
        # The original paper of resnet set total stride=8 with less channels.
        self.layer4 = None
        if not self.cifar_size:
            # print('INFO:PyTorch: Using layer4 for ImageNet Training')
            self.layer4 = self._make_layer(
                block,
                inplanes_origin * 8,
                layers[3],
                stride=strides[3],
                dilate=replace_stride_with_dilation[2],
            )
            inplanes_now = inplanes_origin * 8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(inplanes_now * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    # Ref:
                    # Bag of Tricks for Image Classification with Convolutional Neural Networks, 2018
                    # https://arxiv.org/abs/1812.01187
                    # https://github.com/rwightman/pytorch-image-models/blob
                    # /5966654052b24d99e4bfbcf1b59faae8a75db1fd/timm/models/resnet.py#L293
                    # nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.AvgPool2d(
                        kernel_size=2,
                        stride=stride,
                        ceil_mode=True,
                        padding=0,
                        count_include_pad=False,
                    ),
                    conv1x1(self.inplanes, planes * block.expansion, stride=1),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.base_width,
                previous_dilation,
                norm_layer,
                self.stochastic_depth,
                self.se,
                self.ps_shakedrop[self.u_idx],
            )
        )
        # print(self.ps_shakedrop[self.u_idx])
        self.u_idx += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    stochastic_depth=self.stochastic_depth,
                    apply_se=self.se,
                    p_shakedrop=self.ps_shakedrop[self.u_idx],
                )
            )
            # print(self.ps_shakedrop[self.u_idx])
            self.u_idx += 1

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = F.dropout(x, p=self.resnet_dropout, training=self.training)

        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(arch, block, layers, **kwargs)

    return model


def WRN2810(**kwargs):
    return _resnet(
        "wide_resnet28_10",
        BasicBlock,
        [4, 4, 4, 4],
        width_per_group=64,
        num_classes=kwargs["num_classes"],
        hypparams=kwargs,
    )  # , **kwargs)


"""class WRN2810(BaseModel):

    def __init__(self, **kwargs):
        super(WRN2810, self).__init__(**kwargs)

        self.model = _resnet('wide_resnet28_10', BasicBlock, [4, 4, 4, 4],
                   width_per_group=64, num_classes=kwargs["num_classes"], hypparams=kwargs)

    def forward(self, x):
        out = self.model(x)
        return out"""
