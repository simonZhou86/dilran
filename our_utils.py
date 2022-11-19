# helper functions for the project

# Author: Simon Zhou, last modify Nov. 15, 2022

'''
Change log: 
- Simon: file created, implement edge detector
- Simon: create helper function for perceptual loss
- Reacher: create fusion strategy function
'''

import torch
import torch.nn as nn
from torchvision.models import vgg16_bn
import numpy as np
from skimage import feature

class PercepHook:
    '''
    Pytorch forward hook for computing the perceptual loss
    without modifying the original VGG16 network
    '''
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()


def edge_detector(img, sigma):
    '''
    canny edge detection for input image
    
    two choices: 1) edge detection in the training process, 2) not include in training process
    '''
    if len(img.shape) == 3:
        img = img.squeeze(0) # change shape to [256,256]
    
    edges = feature.canny(img, sigma = sigma)

    return edges


def l2_norm():
    '''
    mse loss (matrix F norm)
    '''
    return


def gradient_loss(fused_img, input_img, device):
    '''
    compute image gradient loss between fused image and input image
    '''

    return None


class percep_loss(nn.Module):
    '''
    compute perceptual loss between fused image and input image
    '''
    def __init__(self, block_idx, device):
        '''
        block_index: the index of the block in VGG16 network, int or list
        int represents single layer perceptual loss
        list represents multiple layers perceptual loss
        '''
        super(percep_loss).__init__()
        self.block_idx = block_idx
        self.device = device
        # load vgg16_bn model features
        self.vgg = vgg16_bn(pretrained=True).features.to(device).eval()
        self.loss = nn.MSELoss()

        # unable gradient update
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # remove maxpooling layer and relu layer
        # TODO:check this part on whether we want relu or not
        bns = [i - 2 for i, m in enumerate(self.vgg) if isinstance(m, nn.MaxPool2d)]

        # register forward hook
        self.hooks = [PercepHook(self.vgg[bns[i]]) for i in block_idx]
        self.features = self.vgg[0: bns[block_idx[-1]] + 1]

    def forward(self, inputs, targets):
        '''
        compute perceptual loss between inputs and targets
        '''
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        assert len(input_features) == len(target_features), 'number of input features and target features should be the same'
        loss = 0
        for i in range(len(input_features)):
            loss += self.loss(input_features[i], target_features[i]) # mse loss
        
        return loss


def compute_perp_loss():
    '''
    you can use the perp_loss class to compute perceptual loss
    '''
    return None


def l1_norm(matrix):
    return torch.abs(matrix).sum()


def fusion_strategy(f1, f2, strategy="average"):
    """
    f1: the extracted features of images 1
    f2: the extracted features of images 2
    strategy: 6 fusion strategy, including:
    "addition", "average", "FER", "L1NW", "AL1NW", "FL1N"
    addition strategy
    average strategy
    FER strategy: Feature Energy Ratio strategy
    L1NW strategy: L1-Norm Weight Strategy
    AL1NW strategy: Average L1-Norm Weight Strategy
    FL1N strategy: Feature L1-Norm Strategy

    Note:
    If the original image is PET or SPECT modal,
    it should be converted into YCbCr data, including Y1, Cb and Cr.
    """

    # The fused feature
    fused = np.zeros_like(f1)
    if strategy == "addition":
        fused = f1 + f2
    elif strategy == "average":
        fused = (f1 + f2) / 2
    elif strategy == "FER":
        k1 = f1 ** 2 / (f1 ** 2 + f2 ** 2)
        k2 = f2 ** 2 / (f1 ** 2 + f2 ** 2)
        fused = k1 * f1 + k2 * f2
    elif strategy == "L1NW":
        l1 = l1_norm(f1)
        l2 = l1_norm(f2)
        fused = l1 * f1 + l2 * f2
    elif strategy == "AL1NW":
        p1 = l1_norm(f1) / 2
        p2 = l1_norm(f2) / 2
        fused = p1 * f1 + p2 * f2
    elif strategy == "FL1N":
        l1 = l1_norm(f1)
        l2 = l1_norm(f2)
        w1 = l1 / (l1 + l2)
        w2 = l2 / (l1 + l2)
        fused = w1 * f1 + w2 * f2
    # Need to do reconstruction on "fused"
    return fused
