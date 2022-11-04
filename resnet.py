# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 10:46
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : resnet.py
# @Software: PyCharm


import torch
from torch import nn
import torchinfo
import onnx
import netron


class ConvBN(nn.Module):
    """定义conv+bn结构，可选relu"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, do_relu=False):
        super(ConvBN, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels)]
        if do_relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv_bn_relu = nn.Sequential(*layers)

    def forward(self, in_lyr):
        return self.conv_bn_relu(in_lyr)


class BasicBlock(nn.Module):
    """定义BasicBlock结构"""

    def __init__(self, in_channels, out_channels, stride=None, downsample=None):
        super(BasicBlock, self).__init__()
        # 是否进行降采样
        self.downsample = downsample
        # 残差计算后的relu
        self.relu = nn.ReLU(inplace=True)
        # 定义主路：conv+bn+relu+conv+bn
        layers = [ConvBN(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, do_relu=True),
                  ConvBN(out_channels, out_channels, kernel_size=3, stride=1, padding=1, do_relu=False)]
        self.basicblock = nn.Sequential(*layers)

        # 是否进行降采样
        if downsample:
            # 定义降采样旁路：1×1conv
            self.do_downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, in_lyr):
        # 进行主路计算
        basicblock_lyr = self.basicblock(in_lyr)
        if self.downsample:
            # 若进行降采样，通过降采样旁路进行计算，再相加
            shortcut_lyr = self.do_downsample(in_lyr)
            residual_lyr = self.relu(basicblock_lyr + shortcut_lyr)
        else:
            # 若不进行降采样，直接进行相加
            residual_lyr = self.relu(basicblock_lyr + in_lyr)
        return residual_lyr  # 返回残差块BasicBlock


class BottleNeck(nn.Module):
    """定义BottleNeck结构"""
    def __init__(self, in_channels, mid_channels, out_channels, stride=None, downsample=None):
        super(BottleNeck, self).__init__()
        # 是否进行降采样
        self.downsample = downsample
        # 残差计算后的relu
        self.relu = nn.ReLU(inplace=True)
        # 定义主路
        layers = [ConvBN(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, do_relu=True),
                  ConvBN(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, do_relu=True),
                  ConvBN(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, do_relu=False)]
        self.bottleneck = nn.Sequential(*layers)
        # 是否进行降采样
        if downsample:
            self.do_downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, in_lyr):
        # 进行主路计算
        bottleneck_lyr = self.bottleneck(in_lyr)
        if self.downsample:
            # 若进行降采样，通过降采样旁路进行计算，再相加
            shortcut_lyr = self.do_downsample(in_lyr)
            residual_lyr = self.relu(bottleneck_lyr + shortcut_lyr)
        else:
            # 若不进行降采样，直接进行相加
            residual_lyr = self.relu(bottleneck_lyr + in_lyr)
        return residual_lyr


class Resnet_bb(nn.Module):
    """创建基于BasicBlock的Resnet18或34"""

    def __init__(self, in_channels, conv_num=None, class_num=None):
        super(Resnet_bb, self).__init__()
        # 第一组卷积组合
        self.conv1 = ConvBN(in_channels, 64, kernel_size=7, stride=2, padding=3, do_relu=True)
        # 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 根据传入的残差块层数，创建残差块的组合
        def get_conv_block(in_channels, out_channels, conv_count=None, downsample=None):
            if downsample:
                convblock = [BasicBlock(in_channels, out_channels, stride=2, downsample=True)]
            else:
                convblock = [BasicBlock(in_channels, out_channels, stride=1, downsample=False)]

            for _ in range(conv_count - 1):
                convblock.append(BasicBlock(out_channels, out_channels, stride=1, downsample=False))

            return convblock

        self.conv2 = nn.Sequential(*get_conv_block(64, 64, conv_num[0], downsample=False))
        self.conv3 = nn.Sequential(*get_conv_block(64, 128, conv_num[1], downsample=True))
        self.conv4 = nn.Sequential(*get_conv_block(128, 256, conv_num[2], downsample=True))
        self.conv5 = nn.Sequential(*get_conv_block(256, 512, conv_num[3], downsample=True))
        # 平均池化+全连接层
        self.avepool = nn.AvgPool2d(kernel_size=7)
        self.full = nn.Linear(512, class_num)

    def forward(self, in_lyr):
        conv1_lyr = self.conv1(in_lyr)
        maxpool_lyr = self.maxpool(conv1_lyr)
        conv2_lyr = self.conv2(maxpool_lyr)
        conv3_lyr = self.conv3(conv2_lyr)
        conv4_lyr = self.conv4(conv3_lyr)
        conv5_lyr = self.conv5(conv4_lyr)
        avepool_lyr = self.avepool(conv5_lyr)
        full_lyr = self.full(avepool_lyr.view(avepool_lyr.size(0), -1))
        return full_lyr


class Resnet_bn(nn.Module):
    """创建基于BottleNeck的Resnet50或101"""

    def __init__(self, in_channels, conv_num=None, class_num=None):
        super(Resnet_bn, self).__init__()
        # 第一组卷积组合
        self.conv1 = ConvBN(in_channels, 64, kernel_size=7, stride=2, padding=3, do_relu=True)
        # 最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        # 根据传入的残差块层数，创建残差块的组合
        def get_conv_block(in_channels, mid_channels, out_channels, conv_count=None, stride=None):
            convblock = [BottleNeck(in_channels, mid_channels, out_channels, stride=stride, downsample=True)]
            for _ in range(conv_count - 1):
                convblock.append(BottleNeck(out_channels, mid_channels, out_channels, stride=1, downsample=False))

            return convblock

        self.conv2 = nn.Sequential(*get_conv_block(64, 64, 256, conv_num[0], stride=1))
        self.conv3 = nn.Sequential(*get_conv_block(256, 128, 512, conv_num[1], stride=2))
        self.conv4 = nn.Sequential(*get_conv_block(512, 256, 1024, conv_num[2], stride=2))
        self.conv5 = nn.Sequential(*get_conv_block(1024, 512, 2048, conv_num[3], stride=2))
        # 平均池化+全连接层
        self.avepool = nn.AvgPool2d(kernel_size=7)
        self.full = nn.Linear(2048, class_num)

    def forward(self, in_lyr):
        conv1_lyr = self.conv1(in_lyr)
        maxpool_lyr = self.maxpool(conv1_lyr)
        conv2_lyr = self.conv2(maxpool_lyr)
        conv3_lyr = self.conv3(conv2_lyr)
        conv4_lyr = self.conv4(conv3_lyr)
        conv5_lyr = self.conv5(conv4_lyr)
        avepool_lyr = self.avepool(conv5_lyr)
        full_lyr = self.full(avepool_lyr.view(avepool_lyr.size(0), -1))
        return full_lyr


if __name__ == "__main__":
    resnet = Resnet_bn(in_channels=3, conv_num=[3, 4, 6, 3], class_num=1000).cuda()
    print(resnet)
    torchinfo.summary(resnet, input_size=(1, 3, 224, 224))
    modelData = 'resnet_test.pth'
    test = torch.randn([1, 3, 224, 224]).cuda()
    # 保存为onnx格式
    torch.onnx.export(resnet, test, modelData, export_params=True)
    # 重新读取
    onnx_model = onnx.load(modelData)
    # 增加特征图纬度信息
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), modelData)
    # 显示网络结构
    netron.start(modelData)
