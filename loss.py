# Loss functions for the project

# Author: Reacher Z., last modify Nov. 18, 2022

"""
Change log:
- Reacher: file created, implement L1 loss and L2 loss function
"""

import numpy as np
import torch


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
