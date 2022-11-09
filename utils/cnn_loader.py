import os
import SimpleITK as sitk
import torch
import numpy as np

'''
load data for cnn

Last modified: Simon Zhou, 2022/1/2
'''

def sort(string):
    string = string.split('_')
    if string[0] == "PCAD":
        first_number = int(string[1])
    else:
        first_number = int(string[0].split('-')[1])

    position = first_number*10 + int(string[-2])
    return position


def fetch_data(dir, SLICES, HEIGHT, WIDTH):
    os.chdir(dir)
    data = torch.empty(0, 1, SLICES, HEIGHT, WIDTH)
    labels = torch.empty(0)
    kgh = sorted([f for f in os.listdir() if "PCAD" in f], key=sort)
    prostateX = sorted([f for f in os.listdir() if "ProstateX" in f], key=sort)
    sorted_dir = kgh + prostateX

    total_labels = 0
    for _, f in enumerate(sorted_dir):
        crop = sitk.ReadImage(f)
        crop = sitk.GetArrayFromImage(crop).astype(np.float32)
        crop = torch.from_numpy(crop)
        crop = crop.reshape(1, 1, *crop.shape)
        label = f.split('_')[-2]
        total_labels += int(label)
        label = torch.Tensor([int(label)])
        data = torch.cat((data, crop), dim=0)
        labels = torch.cat((labels, label))

    return data, labels

