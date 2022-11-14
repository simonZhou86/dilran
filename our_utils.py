# helper functions for the project

# Author: Simon Zhou, last modify Nov. 13, 2022

'''
Change log: 
- Simon: file created, implement edge detector
'''

import torch
import torch.nn as nn
import numpy as np
from skimage import feature


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


def gradient_loss():
    '''
    image gradient loss
    '''
    return


def percep_loss():
    '''
    perceptual loss
    '''
    return