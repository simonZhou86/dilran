# Validation script for the project
# Validate a trained medical image fusion model
# Author: Reacher, last modify Nov. 28, 2022

'''
Change log:
Reacher: file created
'''

from evaluation_metrics import *


# run validation for every epoch


def run(
        batch_size=32,
        model=None,
        dataloader=None,
        metrics="SSIM",
):
    return None
