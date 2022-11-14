
# Model Architecture
# Author: Landy Xu, created on Nov. 12, 2022
# Last modified by Simon on Nov. 13

'''
Change log: 

- Landy: create feature extractor and DILRAN
- Simon: revise some writing style of module configs (e.g., replace = True),
refine the FE module, add recon module
- 
'''


import torch
import torch.nn as nn
import numpy as np

class DILRAN(nn.Module):
    def __init__(self):
        super(DILRAN, self).__init__()
        # TODO: confirm convolution
        self.conv = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.AvgPool2d(2, 2)
        self.lu = nn.ReLU(replace = True)

    def forward(self, x):
        prev = self.conv(x) + self.conv(self.conv(x)) + self.conv(self.conv(self.conv(x)))
        return torch.mul(self.lu(self.up(self.down(x))), prev) + x


class FeatureExtractor(nn.Module):
    def __init__(self, level):
        super(FeatureExtractor, self).__init__()
        # TODO: confirm dilated convolution
        self.conv = nn.Conv2d(1, 64, (1, 1), (1, 1), (0, 0), dilation = 2)
        self.network = DILRAN()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.AvgPool2d(2, 2)
        self.lu = nn.ReLU(replace = True)

    def forward(self, x):
        n1 = self.network(self.conv(x[0]))
        n2 = self.network(self.conv(x[1]))
        n3 = self.network(self.conv(x[2]))
        return torch.cat((n1, n2, n3), 0)


class DILRAN_V1(nn.Module):
    '''
    V1: concat the output of three (conv-d,DILRAN) paths channel wise and add the low level feature to the concat output
    temporary, will edit if necessary
    '''
    def __init__(self, cat_first = False, use_leaky = False):
        super(DILRAN_V1, self).__init__()
        # cat_first, whether to perform channel-wise concat before DILRAN
        # convolution in DILRAN, in channel is the channel from the previous block
        if not cat_first:
            self.conv_d = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same")
            self.bnorm = nn.BatchNorm2d(num_features=64)
        else:
            self.conv_d = nn.Conv2d(in_channels=64*3, out_channels=64*3, kernel_size=3, stride=1, padding="same")
            self.bnorm = nn.BatchNorm2d(num_features=64*3)
        
        if not use_leaky:
            self.relu = nn.ReLU(inplace = True)
        else:
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        self.down = nn.AvgPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    
    def forward(self, x):
        # pooling -> upsample -> ReLU block
        pur_path = self.relu(self.up(self.down(x)))
        # 3*3, 5*5, 7*7 multiscale addition block
        conv_path = self.conv_d(x) + self.conv_d(self.conv_d(x)) + self.conv_d(self.conv_d(self.conv_d(x)))
        # attention
        attn = torch.mul(pur_path, conv_path)
        # residual + attention
        resid_x = x + attn
        return resid_x


class FE_V1(nn.Module):
    '''
    feature extractor block (temporary, will edit if necessary)
    '''
    def __init__(self):
        super(FE_V1, self).__init__()

        # multiscale dilation conv2d
        self.convd1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding="same")
        self.convd2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=3, padding="same")
        self.convd3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, dilation=5, padding="same")

        self.relu = nn.ReLU(inplace = True)
        self.lrelu = nn.LeakyReLU(0.2, inplace = True)

        self.bnorm1 = nn.BatchNorm2d(num_features=64)
        
        self.dilran = DILRAN_V1()

    
    def forward(self, x):
        # dilated convolution
        dilf1 = self.convd1(x)
        dilf2 = self.convd2(x)
        dilf3 = self.convd3(x)

        # DILRAN
        dilran_o1 = self.dilran(dilf1)
        # batchnorm
        dilran_o1 = self.bnorm1(dilran_o1)
        dilran_o2 = self.dilran(dilf2)
        # batchnorm
        dilran_o2 = self.bnorm1(dilran_o2)
        dilran_o3 = self.dilran(dilf3)
        # batchnorm
        dilran_o3 = self.bnorm1(dilran_o3)
        # concat
        cat_o = torch.cat((dilran_o1, dilran_o2, dilran_o3), dim = 1) # first dim is batch, second dim is channel

        return cat_o

class MSFuNet(nn.Module):
    '''
    the whole network (from input image -> feature maps to be used in fusion strategy)
    temporary, will edit if necessary
    '''
    def __init__(self):
        super(MSFuNet, self).__init__()
    
        self.conv_id = nn.Conv2d(in_channels=64, out_channels=64*3, kernel_size=1, stride=1, padding="valid")
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding="same"),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64*3, out_channels=128, kernel_size=3, stride=1, padding="same"),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="same"),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(inplace=True))

        self.lrelu = nn.LeakyReLU(0.2, inplace = True)

        self.fe = FE_V1()

    def forward(self, x):
        x = self.conv1(x) # shallow feature

        # feature returned from feature extractor
        cat_feature = self.fe(x)
        # short cut connection 
        expand_x = self.conv_id(x)
        add = expand_x + cat_feature

        add = self.conv2(add)
        add = self.conv3(add) # should get shape [b, 64, 256, 256]
        return add


class Recon(nn.Module):
    '''
    reconstruction module (temporary, will edit if necessary)
    '''
    def __init__(self):
        super(Recon, self).__init__()

        self.recon_conv = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding="same"),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding="same"),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding="same"),
                                        nn.ReLU(inplace=True))
        
    def forward(self, x):
        x = self.recon_conv(x)
        return x # should get shape [b, 1, 256, 256]
        
