import torch
import torch.nn as nn
from .spikes import *
from .t2c.methods import QConv2d
from models.layer import ConvWithMask

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x

def conv_mask(in_planes, out_planes, stride=1, groups=1, dilation=1, counter=0):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)
    batchnorm = nn.BatchNorm2d(out_planes)
    act = nn.ReLU()
    return ConvWithMask(conv, batchnorm, act, target = -1, nclass=10, ratio=0.85, layerid = counter, do_softmax = 1, mode='decoupled', gt_type='mass').to('cuda')

def qconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, wbit=2, abit=32):
    """3x3 convolution with padding"""
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation, wbit=wbit, abit=abit)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
                base_width=64, dilation=1, norm_layer=None, lb=-3.0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = tdBatchNorm
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.convmask1 = conv_mask(inplanes, planes, stride)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.convmask2 = conv_mask(planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.conv2_s = tdLayer(self.conv2, self.bn2)

        self.spike1 = QLIFSpike(lb=lb)
        self.spike2 = QLIFSpike(lb=lb)
        # self.spike1 = LIFSpike()
        # self.spike2 = LIFSpike()

    def forward(self, x):
        identity = x
        all_mask_logits, all_gt_mask = [], []
        bs = x.shape[0]
        prev = torch.ones(bs) * x.shape[2]
        input_l = x
        # binary_mask = 0
        _, binary_mask ,all_mask_logits, active_cur, all_gt_mask, FLOPs, act_FLOPs = self.convmask1(x, all_mask_logits, all_gt_mask, prev)
        out = self.conv1_s(x)
        # print(x.shape)
        # print(out.shape)
        # import pdb;pdb.set_trace()
        out = self.spike1(out, binary_mask, input_l)

        _, binary_mask ,all_mask_logits, active_cur, all_gt_mask, FLOPs, act_FLOPs = self.convmask2(out, all_mask_logits, all_gt_mask, prev)
        
        out = self.conv2_s(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.spike2(out, binary_mask, input_l)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, lb=-3.0):
        super(ResNet, self).__init__()
        self.flops = 0
        self.actual_flops = 0
        self.active_channels = 0
        self.total_channels = 0
        if norm_layer is None:
            norm_layer = tdBatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 128
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.convmask3 = conv_mask(3, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.conv1_s = tdLayer(self.conv1, self.bn1)
        self.layer1 = self._make_layer(block, 64, layers[0], lb=lb)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, dilate=replace_stride_with_dilation[1], lb=lb)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, dilate=replace_stride_with_dilation[2], lb=lb)
        self.avgpool = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc1 = nn.Linear(512 * block.expansion, 256)
        self.fc1_s = tdLayer(self.fc1)
        self.fc2 = nn.Linear(256, num_classes)
        self.fc2_s = tdLayer(self.fc2)
        self.spike1 = QLIFSpike(lb=lb)
        self.spike = LIFSpike()
        self.T = 1

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, lb=-3.0):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = tdLayer(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, lb=lb))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, lb=lb))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        all_mask_logits, all_gt_mask = [], []
        bs = x.shape[0]
        prev = torch.ones(bs) * x.shape[2]
        input_l = x
        # binary_mask = 0
        out, binary_mask ,all_mask_logits, active_cur, all_gt_mask, FLOPs, act_FLOPs = self.convmask3(x, all_mask_logits, all_gt_mask, prev)
        x = self.conv1_s(x)
        x = self.spike1(x,binary_mask, input_l)
        #print(x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1_s(x)
        x = self.spike(x)
        x = self.fc2_s(x)

        self.flops += FLOPs.mean()
        self.active_channels += active_cur.mean()
        self.total_channels += all_mask_logits[0].size(1)
        self.actual_flops += act_FLOPs.mean()
        return x

    def forward(self, x):
        x = add_dimention(x, self.T)
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, lb, **kwargs):
    model = ResNet(block, layers, lb=lb, **kwargs)
    return model

def resnet19(pretrained=False, progress=True, lb=-3.0, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [3, 3, 2], pretrained, progress, lb=lb, **kwargs)
