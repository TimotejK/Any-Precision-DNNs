import torch
import torch.nn as nn
from .quan_ops import conv2d_quantize_fn, activation_quantize_fn, batchnorm_fn

__all__ = ['mobileNetV2s']

from .resnet_quan import Activate
import math
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, bit_list, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.bit_list = bit_list

        assert stride in [1, 2]

        norm_layer = batchnorm_fn(self.bit_list)
        Conv2d = conv2d_quantize_fn(self.bit_list)

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                Conv2d(inp, expand_inp, 1, 1, 0, bias=False),
                norm_layer(expand_inp),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            Conv2d(
                expand_inp, expand_inp, 3, stride, 1,
                groups=expand_inp, bias=False),
            norm_layer(expand_inp),
            nn.ReLU6(inplace=True),
            Conv2d(expand_inp, outp, 1, 1, 0, bias=False),
            norm_layer(outp),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class MobileNetV2(nn.Module):
    def __init__(self, bit_list, num_classes=1000, input_size=224):
        super(MobileNetV2, self).__init__()
        self.bit_list = bit_list

        norm_layer = batchnorm_fn(self.bit_list)
        Conv2d = conv2d_quantize_fn(self.bit_list)

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.features = []

        # head
        assert input_size % 32 == 0
        channels = 32
        self.outp = 1280
        first_stride = 2
        self.features.append(
            nn.Sequential(
                Conv2d(
                    3, channels, 3,
                    first_stride, 1, bias=False),
                norm_layer(channels),
                nn.ReLU6(inplace=True))
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(bit_list, channels, outp, s, t))
                else:
                    self.features.append(
                        InvertedResidual(bit_list, channels, outp, 1, t))
                channels = outp

        # tail
        self.features.append(
            nn.Sequential(
                Conv2d(
                    channels,
                    self.outp,
                    1, 1, 0, bias=False),
                nn.BatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.outp)
        x = self.classifier(x)
        return x


def mobileNetV2s(bit_list, number_of_classes):
    init_block_channels = 32

    net = MobileNetV2(
        bit_list,
        input_size=init_block_channels,
        num_classes=number_of_classes)

    return net