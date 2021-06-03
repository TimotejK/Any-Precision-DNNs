from pytorchcv.model_provider import get_model as ptcv_get_model
from torch import tensor, nn


class ConvSettings(nn.Conv2d):
    # def __init__(self, groups, weight, bias, stride, padding, dilation, in_channels, out_channels, kernel_size):
    #
    #     self.groups = groups
    #     self.weight = weight
    #     self.bias = bias
    #     if self.bias is None:
    #         self.bias = tensor([0.0] * in_channels)
    #     self.stride = stride
    #     self.padding = padding
    #     self.dilation = dilation
    #     self.in_channels = in_channels
    #     self.out_channels = out_channels
    #     self.kernel_size = kernel_size
    def __init__(self, *kargs, **kwargs):
        super(ConvSettings, self).__init__(*kargs, **kwargs)

def get_conv_settings(conv):
    return ConvSettings(groups=conv.groups,
                 weight=conv.weight,
                 bias=conv.bias,
                 stride=conv.stride,
                 padding=conv.padding,
                 dilation=conv.dilation,
                 in_channels=conv.in_channels,
                 out_channels=conv.out_channels,
                 kernel_size=conv.kernel_size)