import torch
import torch.nn as nn
import numpy as np


class FeatureExtraction(nn.Module):
    def __init__(self, level):
        super(FeatureExtraction, self).__init__()
        self.level = level
        self.conv0 = nn.Conv2d(1, 64, (1, 1), (1, 1), (0, 0))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.AvgPool2d(2, 2)
        self.lu = nn.ReLU()
        self.block = block()

    def forward(self, x):
        tem = self.conv0(x)
        a = torch.mul(self.lu(self.up(self.down(x))), tem) + x
        tem = self.block(tem)
        tem = self.block(tem)
        tem = self.block(tem)
        return torch.mul(self.lu(self.up(self.down(a))), tem) + a


class block(nn.Module):
    def __init__(self):
        super(block, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.AvgPool2d(2, 2)
        self.lu = nn.ReLU()
        self.norm = nn.BatchNorm2d(64)

    def forward(self, x):
        p0 = self.down(x)
        put0 = self.conv1(p0) + self.conv1(self.conv1(p0)) + self.conv1(self.conv1(self.conv1(p0)))
        out0 = torch.mul(self.lu(self.up(self.down(p0))), put0) + p0
        p1 = self.down(p0)
        put1 = self.conv1(p1) + self.conv1(self.conv1(p1)) + self.conv1(self.conv1(self.conv1(p1)))
        out1 = torch.mul(self.lu(self.up(self.down(p1))), put1) + p1
        p2 = self.down(p1)
        put2 = self.conv1(p2) + self.conv1(self.conv1(p2)) + self.conv1(self.conv1(self.conv1(p2)))
        out2 = torch.mul(self.lu(self.up(self.down(p2))), put2) + p2

        out2 = self.up(out2)
        out1 = out1 + out2
        out1 = self.up(out1)
        out0 = out0 + out1
        out0 = self.up(out0)
        out = torch.mul(self.lu(self.up(self.down(x))), out0) + x
        out = self.norm(out)
        return out


class ImageReconstruction(nn.Module):
    def __init__(self):
        super(ImageReconstruction, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv1 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(32, 1, (1, 1), (1, 1), (0, 0))

    def forward(self, x):
        out0 = self.conv0(x)
        out1 = self.conv1(out0)
        out2 = self.conv2(out1)
        return out2


class SRN(nn.Module):
    def __init__(self):
        super(SRN, self).__init__()

        self.fe = FeatureExtraction(level=3)
        self.recon = ImageReconstruction()

    def forward(self, LR):
        deep_fe = self.fe(LR)
        recon_img = self.recon(deep_fe)

        return recon_img
