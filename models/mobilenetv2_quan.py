import torch
import torch.nn as nn
from utils.quant_modules import QuantBnConv2d, QuantAct, QuantConv2d, QuantAveragePool2d
from .quan_ops import conv2d_quantize_fn, activation_quantize_fn, batchnorm_fn

__all__ = ['mobileNetV2']

class LinearBottleneckQ(nn.Module):
    expansion = 4

    def __init__(self, bit_list, in_planes, out_planes, stride=1, expansion=False, remove_exp_conv=False):
        super(LinearBottleneckQ, self).__init__()
        self.bit_list = bit_list
        self.wbit = self.bit_list[-1]
        self.abit = self.bit_list[-1]

        self.residual = (in_planes == out_planes) and (stride == 1)
        mid_channels = in_planes * 6 if expansion else in_planes
        self.use_exp_conv = (expansion or (not remove_exp_conv))
        self.activatition_func = nn.ReLU6()

        norm_layer = batchnorm_fn(self.bit_list)

        self.quant_act = QuantAct()

        if self.use_exp_conv:
            self.bn1 = norm_layer(in_planes)
            self.conv1 = QuantConv2d(in_planes, in_planes, kernel_size=1, stride=1, bias=False, padding=0, groups=1)
            self.quant_act1 = QuantAct()

        self.bn2 = norm_layer(in_planes)
        self.conv2 = QuantConv2d(1, in_planes, kernel_size=3, stride=1, bias=False, padding=1, groups=in_planes)
        self.quant_act2 = QuantAct()

        self.bn3 = norm_layer(in_planes)
        self.conv3 = QuantConv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False, padding=0, groups=1)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        if self.residual:
            identity = x

        x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, None, None, None, None)

        if self.use_exp_conv:
            x, weight_scaling_factor = self.conv1(self.bn1(x), act_scaling_factor)
            x = self.activatition_func(x)
            x, self.act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None)

            x, weight_scaling_factor = self.conv2(self.bn2(x), act_scaling_factor)
            x = self.activatition_func(x)
            x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

            # note that, there is no activation for the last conv
            x, weight_scaling_factor = self.conv3(self.bn3(x), act_scaling_factor)
        else:
            x, weight_scaling_factor = self.conv2(self.bn2(x), act_scaling_factor)
            x = self.activatition_func(x)
            x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

            # note that, there is no activation for the last conv
            x, weight_scaling_factor = self.conv3(self.bn3(x), act_scaling_factor)

        if self.residual:
            x = x + identity
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity,
                                                         scaling_factor_int32, None)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None, None)

        return x, act_scaling_factor


class Q_MobileNetV2(nn.Module):
    """
    Quantized MobileNetV2 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    remove_exp_conv : bool
        Whether to remove expansion convolution.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 bit_list,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 remove_exp_conv,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(Q_MobileNetV2, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.channels = channels
        self.activatition_func = nn.ReLU6()

        # add input quantization
        self.quant_input = QuantAct()

        # change the inital block
        self.add_module("init_block", QuantConv2d(in_channels=in_channels, out_channels=init_block_channels, kernel_size=3, stride=2, padding=1, groups=1, bias=False))

        # self.init_block.set_param(model.features.init_block.conv, model.features.init_block.bn)

        self.quant_act_int32 = QuantAct()

        self.features = nn.Sequential()
        # change the middle blocks
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)

                stage.add_module("unit{}".format(j + 1), LinearBottleneckQ(
                    bit_list,
                    in_planes=in_channels,
                    out_planes=out_channels,
                    stride=stride,
                    expansion=expansion,
                    remove_exp_conv=remove_exp_conv,
                    ))

                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        # change the final block
        self.quant_act_before_final_block = QuantAct()
        self.features.add_module("final_block", QuantConv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, bias=False))

        self.quant_act_int32_final = QuantAct()

        in_channels = final_block_channels

        self.features.add_module("final_pool", QuantAveragePool2d())
        self.quant_act_output = QuantAct()

        self.output = QuantConv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1, stride=1, bias=False, padding=0, groups=1)

    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)

        # the init block
        x, weight_scaling_factor = self.init_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None)

        # the feature block
        for i, channels_per_stage in enumerate(self.channels):
            cur_stage = getattr(self.features, f'stage{i+1}')
            for j, out_channels in enumerate(channels_per_stage):
                cur_unit = getattr(cur_stage, f'unit{j+1}')

                x, act_scaling_factor = cur_unit(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_before_final_block(x, act_scaling_factor, None, None, None, None)
        x, weight_scaling_factor = self.features.final_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        x, act_scaling_factor = self.quant_act_int32_final(x, act_scaling_factor, weight_scaling_factor, None, None, None)

        # the final pooling
        x = self.features.final_pool(x, act_scaling_factor)

        # the output
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, None, None, None, None)
        x, act_scaling_factor = self.output(x, act_scaling_factor)

        x = x.reshape(x.size(0), -1)

        return x



def mobileNetV2(bit_list, number_of_classes, remove_exp_conv=False):
    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 2, 3, 4, 3, 3, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(channels_per_layers, layers, downsample),
        [[]])

    net = Q_MobileNetV2(
        bit_list,
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        remove_exp_conv=remove_exp_conv,
        num_classes=number_of_classes)

    return net