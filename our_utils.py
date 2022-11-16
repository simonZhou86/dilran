# helper functions for the project

# Author: Simon Zhou, last modify Nov. 15, 2022

'''
Change log: 
- Simon: file created, implement edge detector
- Simon: create perceptual loss related func
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
    def __init__(self) -> None:
        super().__init__()
