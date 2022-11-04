# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 10:46
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : unet.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torchinfo
import netron
import onnx


class DoubleConv(nn.Module):
    """定义卷积块（双卷积+relu）"""

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),  # inplace=True,结果直接替换输入的层，节省内存
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, in_channel):
        return self.doubleconv(in_channel)


class DownSampling(nn.Module):
    """定义下采样操作"""

    def __init__(self, in_channel, out_channel):
        super(DownSampling, self).__init__()
        self.down_sampling = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, in_channel):
        return self.down_sampling(in_channel)


class UpSampling(nn.Module):
    """定义上采样操作"""

    def __init__(self, in_channel, out_channel):
        super(UpSampling, self).__init__()
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True),  # inplace=True,结果直接替换输入的层，节省内存
            nn.BatchNorm2d(out_channel)
        )
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, in_lyr, down_lyr):
        lyr_1 = self.up_sampling(in_lyr),
        cat_lyr = torch.cat([lyr_1[0], down_lyr], dim=1)
        out_lyr = self.conv(cat_lyr)
        return out_lyr


class UNet(nn.Module):  # 继承nn.Module类
    """定义Unet网路结构"""

    def __init__(self, in_channel, out_channel, hidden_channels):
        super(UNet, self).__init__()  # 利用父类（nn.Module）的初始化方法来初始化继承的属性

        self.conv_1 = DoubleConv(in_channel, hidden_channels[0])  # 进行两次（卷积+relu+BN）
        self.down_1 = DownSampling(hidden_channels[0], hidden_channels[1])
        self.down_2 = DownSampling(hidden_channels[1], hidden_channels[2])
        self.down_3 = DownSampling(hidden_channels[2], hidden_channels[3])
        self.down_4 = DownSampling(hidden_channels[3], hidden_channels[4])
        self.up_1 = UpSampling(hidden_channels[4], hidden_channels[3])
        self.up_2 = UpSampling(hidden_channels[3], hidden_channels[2])
        self.up_3 = UpSampling(hidden_channels[2], hidden_channels[1])
        self.up_4 = UpSampling(hidden_channels[1], hidden_channels[0])
        self.conv_2 = nn.Conv2d(in_channels=hidden_channels[0], out_channels=out_channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_lyr):
        conv_lyr_1 = self.conv_1(in_lyr)
        down_lyr_1 = self.down_1(conv_lyr_1)
        down_lyr_2 = self.down_2(down_lyr_1)
        down_lyr_3 = self.down_3(down_lyr_2)
        down_lyr_4 = self.down_4(down_lyr_3)
        up_lyr_1 = self.up_1(down_lyr_4, down_lyr_3)
        up_lyr_2 = self.up_2(up_lyr_1, down_lyr_2)
        up_lyr_3 = self.up_3(up_lyr_2, down_lyr_1)
        up_lyr_4 = self.up_4(up_lyr_3, conv_lyr_1)
        conv_lyr_2 = self.conv_2(up_lyr_4)
        return self.sigmoid(conv_lyr_2)


if __name__ == '__main__':
    # hidden_channels = [64, 128, 256, 512, 1024]
    hidden_channels = [32, 64, 128, 256, 512]
    unet = UNet(4, 1, hidden_channels).cuda()
    test = torch.randn([1, 4, 512, 512]).cuda()
    # 打印网络结构
    torchinfo.summary(unet, input_size=(1, 4, 512, 512))
    # 设定网络保存路径
    modelData = 'unet_test.pth'
    # 保存为onnx格式
    torch.onnx.export(unet, test, modelData, export_params=True)
    # 重新读取
    onnx_model = onnx.load(modelData)
    # 增加特征图纬度信息
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), modelData)
    # 显示网络结构
    netron.start(modelData)
