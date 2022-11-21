# Loss functions for the project

# Author: Reacher Z., last modify Nov. 18, 2022

"""
Change log:
- Reacher: file created, implement L1 loss and L2 loss function
- Reacher: update image gradient calculation
"""

import numpy as np
import torch
from our_utils import Percep_loss
from torchmetrics.functional import image_gradients
from torchvision.transforms import transforms


def l1_loss(predicted, target):
    """
    To compute L1 loss using predicted and target
    """
    return torch.abs(predicted - target).mean()


def mse_loss(predicted, target):
    """
    To compute L2 loss between predicted and target
    """
    return torch.pow((predicted - target), 2).mean()


def img_gradient(img):
    """
    Input: one PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
    Output: image gradient (2 x C x H x W)
    """
    trans = transforms.ToTensor()
    # a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    img_tensor = trans(img)
    # reshape to [N, C, H, W]
    img_tensor = img_tensor.reshape((1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2]))
    dy, dx = image_gradients(img_tensor)
    dy, dx = dy.squeeze(), dx.squeeze()
    dxy = torch.stack((dx, dy), axis=0)
    return dxy


def gradient_loss(predicted, target):
    """
    compute image gradient loss between predicted and target
    """
    # grad_p = np.gradient(predicted)
    # grad_t = np.gradient(target)
    grad_p = img_gradient(predicted)
    grad_t = img_gradient(target)
    return torch.pow((grad_p - grad_t), 2).mean()


def perceptual_loss(predicted, target, block_idx, device):
    """
    compute perceptual loss between predicted and target
    """
    p_loss = Percep_loss(block_idx, device)
    return p_loss(predicted, target)


def loss_func(predicted, target, lambda1, lambda2, block_idx, device):
    """
    Implement the loss function in our proposal
    Loss = a variant of the MSE loss + perceptual loss
    """
    loss = mse_loss(predicted, target) + lambda1 * gradient_loss(predicted, target)
    +lambda2 * perceptual_loss(predicted, target, block_idx, device)
    return loss
