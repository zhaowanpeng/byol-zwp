# -*- coding:utf-8 -*-
import torch
import torch.nn as nn



class SDM(nn.Module):

    def __init__(self):
        super(SDM, self).__init__()  # 64*64

        self.extract_1 = Sepres_Block(3, 72, 3)  # 32*32
        self.extract_2 = Sepres_Block(72, 144, 3)  # 16*16
        self.extract_3 = Sepres_Block(144, 288, 3)  # 8*8
        self.merge = Sepres_Block(288, 576, 1)  # 4

    def forward(self, x):
        x = self.extract_1(x)
        x = self.extract_2(x)
        x = self.extract_3(x)
        x = self.merge(x)
        x = x.view(x.size(0), -1)
        return x


class Sepres_Block(nn.Module):

    def __init__(self, in_channel, out_channel, group_num=3):
        super(Sepres_Block, self).__init__()

        self.scope_extract = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=group_num),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=group_num),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, groups=group_num),
            torch.nn.BatchNorm2d(out_channel),
        )

        self.raw_extract = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=group_num),
            torch.nn.BatchNorm2d(out_channel),
        )

        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.scope_extract(x)
        res = self.raw_extract(x)
        out = out + res
        out = self.relu(out)
        out = self.maxpool(out)
        return out