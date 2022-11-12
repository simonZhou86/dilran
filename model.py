'''
Change log: create feature extractor and DILRAN
'''

import torch
import torch.nn as nn


class DILRAN(nn.Module):
    def __init__(self):
        super(DILRAN, self).__init__()
        # TODO: confirm convolution
        self.conv = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.AvgPool2d(2, 2)
        self.lu = nn.ReLU()

    def forward(self, x):
        prev = self.conv(x) + self.conv(self.conv(x)) + self.conv(self.conv(self.conv(x)))
        return torch.mul(self.lu(self.up(self.down(x))), prev) + x


class FeatureExtractor(nn.Module):
    def __init__(self, level):
        super(FeatureExtractor, self).__init__()
        # TODO: confirm dilated convolution
        self.conv = nn.Conv2d(1, 64, (1, 1), (1, 1), (0, 0), dilation = 2)
        self.network = DILRAN()

    def forward(self, x):
        n1 = self.network(self.conv(x[0]))
        n2 = self.network(self.conv(x[1]))
        n3 = self.network(self.conv(x[2]))
        return torch.cat((n1, n2, n3), 0)
        

