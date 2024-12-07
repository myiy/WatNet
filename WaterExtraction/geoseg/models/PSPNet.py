# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import OrderedDict
#
# class _ConvBatchNormReLU(nn.Sequential):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, relu=True):
#         super(_ConvBatchNormReLU, self).__init__()
#         self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                                           stride=stride, padding=padding, dilation=dilation, bias=False))
#         self.add_module("bn", nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.95))
#         if relu:
#             self.add_module("relu", nn.ReLU(inplace=True))
#
# class _Bottleneck(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels, stride, dilation, downsample):
#         super(_Bottleneck, self).__init__()
#         self.reduce = _ConvBatchNormReLU(in_channels, mid_channels, 1, stride=1, padding=0, dilation=1)
#         self.conv3x3 = _ConvBatchNormReLU(mid_channels, mid_channels, 3, stride=stride, padding=dilation, dilation=dilation)
#         self.increase = _ConvBatchNormReLU(mid_channels, out_channels, 1, stride=1, padding=0, dilation=1, relu=False)
#         self.downsample = downsample
#         if self.downsample:
#             self.proj = _ConvBatchNormReLU(in_channels, out_channels, 1, stride=stride, padding=0, dilation=1, relu=False)
#
#     def forward(self, x):
#         out = self.reduce(x)
#         out = self.conv3x3(out)
#         out = self.increase(out)
#         if self.downsample:
#             out += self.proj(x)
#         else:
#             out += x
#         return F.relu(out, inplace=True)
#
# class _ResBlock(nn.Sequential):
#     def __init__(self, n_layers, in_channels, mid_channels, out_channels, stride, dilation):
#         super(_ResBlock, self).__init__()
#         self.add_module("block1", _Bottleneck(in_channels, mid_channels, out_channels, stride, dilation, True))
#         for i in range(2, n_layers + 1):
#             self.add_module("block" + str(i), _Bottleneck(out_channels, mid_channels, out_channels, 1, dilation, False))
#
# class _DilatedFCN(nn.Module):
#     def __init__(self, n_blocks):
#         super(_DilatedFCN, self).__init__()
#         self.layer1 = _ConvBatchNormReLU(3, 64, 7, stride=2, padding=3, dilation=1)
#         self.layer2 = _ResBlock(n_blocks[0], 64, 64, 256, 1, 1)
#         self.layer3 = _ResBlock(n_blocks[1], 256, 128, 512, 2, 1)
#         self.layer4 = _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2)
#         self.layer5 = _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4)
#
#     def forward(self, x):
#         h = self.layer1(x)
#         h = self.layer2(h)
#         h = self.layer3(h)
#         h1 = self.layer4(h)
#         h2 = self.layer5(h1)
#         return h1, h2
#
# class PSPNet(nn.Module):
#     def __init__(self, num_classes=4, n_blocks=[3, 4, 23, 3], input_size=(1024, 1024)):
#         super(PSPNet, self).__init__()
#         self.fcn = _DilatedFCN(n_blocks)
#         # Assume input_size is a tuple (H, W)
#         self.up = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
#         # Followed by other layers, including the Pyramid Pooling Module and final layers
#
#     def forward(self, x):
#         aux, h = self.fcn(x)
#         # Apply upsampling on `h` to match the input size
#         h = self.up(h)
#         # Continue with pyramid pooling module, final convolution, etc.
#         return h  # or return aux, h based on your requirements
# #
# # # Example of creating PSPNet
# # # model = PSPNet(num_classes=8, n_blocks=[3, 4, 23, 3], input_size=(1024, 1024))
# #
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1,
                 bn_momentum=0.0003):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dilation=[1, 1, 1, 1], bn_momentum=0.0003, is_fpn=False):
        self.inplanes = 128
        self.is_fpn = is_fpn
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(64, momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(128, momentum=bn_momentum)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilation=dilation[0], bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1 if dilation[1] != 1 else 2, dilation=dilation[1],
                                       bn_momentum=bn_momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1 if dilation[2] != 1 else 2, dilation=dilation[2],
                                       bn_momentum=bn_momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1 if dilation[3] != 1 else 2, dilation=dilation[3],
                                       bn_momentum=bn_momentum)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1, bn_momentum=0.0003):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=True, momentum=bn_momentum))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                                bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def forward(self, x, start_module=1, end_module=5):
        if start_module <= 1:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.maxpool(x)
            start_module = 2
        features = []
        for i in range(start_module, end_module + 1):
            x = eval('self.layer%d' % (i - 1))(x)
            features.append(x)

        if self.is_fpn:
            if len(features) == 1:
                return features[0]
            else:
                return tuple(features)
        else:
            return x


def get_resnet101(dilation=[1, 1, 1, 1], bn_momentum=0.0003, is_fpn=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3], dilation=dilation, bn_momentum=bn_momentum, is_fpn=is_fpn)
    return model

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class PyramidPooling(nn.Module):
    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6],
                 norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([
                ('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)),
                ('{}/cbr'.format(name),
                 ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True,
                            has_relu=True, has_bias=False,
                            norm_layer=norm_layer))
            ])))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv6 = nn.Sequential(
            ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, out_planes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pooling in self.ppm:
            ppm_out.append(
                F.interpolate(pooling(x), size=(input_size[2], input_size[3]),
                              mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)

        ppm_out = self.conv6(ppm_out)
        return ppm_out

class PSPNet(nn.Module):
    def __init__(self, class_num, bn_momentum=0.01):
        super(PSPNet, self).__init__()
        self.Resnet101 = get_resnet101(dilation=[1, 1, 1, 2], bn_momentum=bn_momentum, is_fpn=False)
        self.psp_layer = PyramidPooling('psp', class_num, 2048, norm_layer=nn.BatchNorm2d)

    def forward(self, input):
        b, c, h, w = input.shape
        x = self.Resnet101(input)
        psp_fm = self.psp_layer(x)
        pred = F.interpolate(psp_fm, size=input.size()[2:4], mode='bilinear', align_corners=True)

        return pred

if __name__ == '__main__':
    from thop import profile
    x = torch.randn(1, 3, 512, 512)
    net = PSPNet()
    out = net(x)
    print(net)
    print(out.shape)
    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)