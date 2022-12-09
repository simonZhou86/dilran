# Evaluation Metrics and get results

# Author: Reacher Z., last modify Nov. 26, 2022

"""
Change log:
- Reacher: file created, implement PSNR, SSIM, NMI, MI
"""

import numpy as np
import sklearn.metrics as skm
import torch
from skimage.metrics import peak_signal_noise_ratio, normalized_mutual_information
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
#from TMQI import TMQI, TMQIr

def psnr(img_pred: torch.Tensor, img_true: torch.Tensor):
    """
    To compute PeakSignalNoiseRatio
    Return: float
    """
    peakSignalNoiseRatio = PeakSignalNoiseRatio(data_range=1.0)
    return peakSignalNoiseRatio(img_pred, img_true).item()


def ssim(img_pred: torch.Tensor, img_true: torch.Tensor):
    """
    To compute PeakSignalNoiseRatio
    Input: [N, C, H, W] shape
    Return: float
    """
    img_pred = img_pred.unsqueeze(0).unsqueeze(0)
    img_true = img_true.unsqueeze(0).unsqueeze(0)
    structuralSimilarityIndexMeasure = StructuralSimilarityIndexMeasure(data_range=1.0)
    return structuralSimilarityIndexMeasure(img_pred, img_true).item()


def nmi(img_pred: torch.Tensor, img_true: torch.Tensor):
    """
    normalized mutual information (NMI)
    Return: float
    """
    img_pred_np = np.array(img_pred.squeeze())
    img_true_np = np.array(img_true.squeeze())
    nor_mi = normalized_mutual_information(img_pred_np, img_true_np)
    return nor_mi


def mutual_information(img_pred: torch.Tensor, img_true: torch.Tensor):
    """
    Mutual Information:
    I(A,B) = H(A) + H(B) - H(A,B)
    H(A)= -sum p(a_i) * log p(a_i)
    Mutual information is a measure of image matching, that does not require the signal
    to be the same in the two images. It is a measure of how well you can predict the signal
    in the second image, given the signal intensity in the first.
    Return: float
    """
    img_pred_uint8 = (np.array(img_pred.squeeze()) * 255).astype(np.uint8).flatten()
    img_true_uint8 = (np.array(img_true.squeeze()) * 255).astype(np.uint8).flatten()
    size = img_true_uint8.shape[-1]
    pa = np.histogram(img_pred_uint8, 256, (0, 255))[0] / size
    pb = np.histogram(img_true_uint8, 256, (0, 255))[0] / size
    ha = -np.sum(pa * np.log(pa + 1e-20))
    hb = -np.sum(pb * np.log(pb + 1e-20))

    pab = (np.histogram2d(img_pred_uint8, img_true_uint8, 256, [[0, 255], [0, 255]])[0]) / size
    hab = -np.sum(pab * np.log(pab + 1e-20))
    mi = ha + hb - hab
    # hist_2d, x_edges, y_edges = np.histogram2d(img_pred.numpy().ravel(), img_true.numpy().ravel(), bins=256)
    # pxy = hist_2d / float(np.sum(hist_2d))
    # px = np.sum(pxy, axis=1) # marginal for x over y
    # py = np.sum(pxy, axis=0) # marginal for y over x
    # px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # # Now we can do the calculation using the pxy, px_py 2D arrays
    # nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    # return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi

def mi2(x, y):
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)
    return skm.mutual_info_score(x, y)

