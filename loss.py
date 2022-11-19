# Loss functions for the project

# Author: Reacher Z., last modify Nov. 18, 2022

"""
Change log:
- Reacher: file created, implement L1 loss and L2 loss function
"""

import numpy as np
import torch
from our_utils import Percep_loss


def l1_loss(predicted, target):
    """
    To compute L1 loss using predicted and target
    """
    return torch.abs(predicted - target).mean()


def mse_loss(predicted, target):
    """
    To compute L2 loss using predicted and target
    """
    return torch.pow((predicted - target), 2).mean()


def gradient_loss(predicted, target):
    """
    compute image gradient loss between fused image and input image
    """
    grad_p = np.gradient(predicted)
    grad_t = np.gradient(target)
    return torch.pow((grad_p - grad_t), 2).mean()


def perceptual_loss(predicted, target, block_idx, device):
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
