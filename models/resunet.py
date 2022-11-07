# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 11:40
# @Author  : Zph
# @Email   : hhs_zph@mail.imu.edu.cn
# @File    : resunet.py
# @Software: PyCharm

import torch
from torch import nn
import torchinfo
import onnx
import netron

class ResBlock(nn.Module):
    """定义ResBlock结构"""

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        # 定义残差结构主路
        res_layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                      nn.BatchNorm2d(out_channels)]
        self.res_layers = nn.Sequential(*res_layers)
        # 定义残差结构旁路
        shortcut_layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)]
        self.shortcut = nn.Sequential(*shortcut_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, in_lyr):
        res = self.res_layers(in_lyr)
        shortcut = self.shortcut(in_lyr)
        return self.relu(res + shortcut)

class DownSampling(nn.Module):
    """定义下采样操作"""

    def __init__(self, in_channel, out_channel):
        super(DownSampling, self).__init__()
        self.down_sampling = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channel, out_channel)
        )

    def forward(self, in_channel):
        return self.down_sampling(in_channel)


class UpSampling(nn.Module):
    """定义上采样操作"""

    def __init__(self, in_channel, out_channel):
        super(UpSampling, self).__init__()
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, bias=False),
            nn.ReLU(inplace=True),  # inplace=True,结果直接替换输入的层，节省内存
            nn.BatchNorm2d(out_channel)
        )
        self.conv = ResBlock(in_channel, out_channel)

    def forward(self, in_lyr, down_lyr):
        lyr_1 = self.up_sampling(in_lyr),
        cat_lyr = torch.cat([lyr_1[0], down_lyr], dim=1)
        out_lyr = self.conv(cat_lyr)
        return out_lyr


class ResUNet(nn.Module):  # 继承nn.Module类
    """定义ResUNet网路结构"""

    def __init__(self, in_channel, out_channel, hidden_channels):
        super(ResUNet, self).__init__()  # 利用父类（nn.Module）的初始化方法来初始化继承的属性

        self.conv_1 = ResBlock(in_channel, hidden_channels[0])  # 进行两次（卷积+relu+BN）
        self.down_1 = DownSampling(hidden_channels[0], hidden_channels[1])
        self.down_2 = DownSampling(hidden_channels[1], hidden_channels[2])
        self.down_3 = DownSampling(hidden_channels[2], hidden_channels[3])
        self.down_4 = DownSampling(hidden_channels[3], hidden_channels[4])

        self.up_1 = UpSampling(hidden_channels[4], hidden_channels[3])
        self.up_2 = UpSampling(hidden_channels[3], hidden_channels[2])
        self.up_3 = UpSampling(hidden_channels[2], hidden_channels[1])
        self.up_4 = UpSampling(hidden_channels[1], hidden_channels[0])

        self.conv_2 = nn.Conv2d(hidden_channels[0], out_channel, kernel_size=1, stride=1, padding=0)
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
    resunet = ResUNet(4, 1, hidden_channels).cuda()

    torchinfo.summary(resunet, input_size=(1, 4, 256, 256))
    #
    # test = torch.randn([1, 4, 256, 256]).cuda()
    # # 设定网络保存路径
    # modelData = 'resunet_test.pth'
    # # 保存为onnx格式
    # torch.onnx.export(resunet, test, modelData, export_params=True, opset_version=9)
    # # 重新读取
    # onnx_model = onnx.load(modelData)
    # # 增加特征图纬度信息
    # onnx.save(onnx.shape_inference.infer_shapes(onnx_model), modelData)
    # # 显示网络结构
    # netron.start(modelData)




    # res = UpSampling(256, 128)
    # in_lyr = torch.randn(size=(1,256,64,64))
    # down_lyr = torch.randn(size=(1,128,128,128))
    # result = res(in_lyr,down_lyr)
    # print(result.size())

    #torchinfo.summary(res, input_size=(1, 512, 256, 256))

















