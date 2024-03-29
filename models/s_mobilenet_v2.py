import math
import torch.nn as nn


from .slimmable_ops import SwitchableBatchNorm2d, SlimmableConv2d
from .slimmable_ops import make_divisible

__all__ = ['slimmableMobileNetV2']

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio, width_mult_list):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp

        layers = []
        # expand
        expand_inp = [i * expand_ratio for i in inp]
        if expand_ratio != 1:
            layers += [
                SlimmableConv2d(inp, expand_inp, 1, 1, 0, bias=False, width_mult_list=width_mult_list),
                SwitchableBatchNorm2d(expand_inp, width_mult_list),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        layers += [
            SlimmableConv2d(
                expand_inp, expand_inp, 3, stride, 1,
                groups_list=expand_inp, bias=False, width_mult_list=width_mult_list),
            SwitchableBatchNorm2d(expand_inp, width_mult_list),
            nn.ReLU6(inplace=True),
            SlimmableConv2d(expand_inp, outp, 1, 1, 0, bias=False, width_mult_list=width_mult_list),
            SwitchableBatchNorm2d(outp, width_mult_list),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection:
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult_list=[0.35, 0.5, 0.75, 1.0], reset_parameters=True):
        super(Model, self).__init__()

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
        channels = [
            make_divisible(32 * width_mult)
            for width_mult in width_mult_list]
        self.outp = make_divisible(
            1280 * max(width_mult_list)) if max(
                width_mult_list) > 1.0 else 1280
        first_stride = 2
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    [3 for _ in range(len(channels))], channels, 3,
                    first_stride, 1, bias=False, width_mult_list=width_mult_list),
                SwitchableBatchNorm2d(channels, width_mult_list),
                nn.ReLU6(inplace=True))
        )

        # body
        for t, c, n, s in self.block_setting:
            outp = [
                make_divisible(c * width_mult)
                for width_mult in width_mult_list]
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(channels, outp, s, t, width_mult_list))
                else:
                    self.features.append(
                        InvertedResidual(channels, outp, 1, t, width_mult_list))
                channels = outp

        # tail
        self.features.append(
            nn.Sequential(
                SlimmableConv2d(
                    channels,
                    [self.outp for _ in range(len(channels))],
                    1, 1, 0, bias=False, width_mult_list=width_mult_list),
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
        if reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.outp)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def slimmableMobileNetV2(bit_list, number_of_classes):
    init_block_channels = 32

    net = Model(
        input_size=init_block_channels,
        num_classes=number_of_classes,
        width_mult_list=bit_list)

    return net