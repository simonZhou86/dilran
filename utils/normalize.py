import torch
import SimpleITK as sitk
import numpy as np

'''
Normalize an image to [-1,1]

Last modified: 2022/1/6 Simon
'''
def normalize(im):

    mins = [im[idx].min() for idx in range(len(im))]
    maxes = [im[idx].max() for idx in range(len(im))]

    for idx in range(len(im)):
        min_val = mins[idx]
        max_val = maxes[idx]

        if min_val == max_val:
            im[idx] = torch.zeros(im[idx].shape)
        else:
            im[idx] = 2*(im[idx] - min_val)/(max_val - min_val) - 1


def normalize01(im):
    temp = im / 255.
    return temp


