# helper functions for the project

# Author: Simon Zhou, last modify Nov. 13, 2022

'''
Change log: 
- Simon: file created, implement edge detector
- Simon: create helper function for perceptual loss
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
        
        # remove maxpooling layer
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