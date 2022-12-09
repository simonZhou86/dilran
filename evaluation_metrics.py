# Evaluation Metrics and get results

# Author: Reacher Z., last modify Nov. 26, 2022

"""
Change log:
- Reacher: file created, implement PSNR, SSIM, NMI, MI
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, normalized_mutual_information
from scipy.stats import entropy
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from sklearn.metrics import mutual_info_score
# import piq
import cv2
import phasepack.phasecong as pc
import skimage.measure as skm

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
    structuralSimilarityIndexMeasure = StructuralSimilarityIndexMeasure(data_range=1.0)
    return structuralSimilarityIndexMeasure(img_pred, img_true).item()


def nmi(img_pred: torch.Tensor, img_true: torch.Tensor):
    """
    normalized mutual information (NMI)
    Return: float
    """
    img_pred_np = np.array(img_pred)#.squeeze())
    img_true_np = np.array(img_true)#.squeeze())
    nor_mi = normalized_mutual_information(img_pred_np, img_true_np)
    return nor_mi


# def mutual_information(img_pred: torch.Tensor, img_true: torch.Tensor):
#     """
#     Mutual Information:
#     I(A,B) = H(A) + H(B) - H(A,B)
#     H(A)= -sum p(a_i) * log p(a_i)
#     Mutual information is a measure of image matching, that does not require the signal
#     to be the same in the two images. It is a measure of how well you can predict the signal
#     in the second image, given the signal intensity in the first.
#
#     Return: float
#     """
#     img_pred_uint8 = (np.array(img_pred.squeeze()) * 255).flatten()
#     img_true_uint8 = (np.array(img_true.squeeze()) * 255).flatten()
#     size = img_true_uint8.shape[-1]
#     pa = np.histogram(img_pred_uint8, 256, (0, 255))[0] / size
#     pb = np.histogram(img_true_uint8, 256, (0, 255))[0] / size
#     ha = -np.sum(pa * np.log(pa + 1e-20))
#     hb = -np.sum(pb * np.log(pb + 1e-20))
#
#     pab = (np.histogram2d(img_pred_uint8, img_true_uint8, 256, [[0, 255], [0, 255]])[0]) / size
#     hab = -np.sum(pab * np.log(pab + 1e-20))
#     mi = ha + hb - hab
#     return mi


def mutual_information(img_pred: torch.Tensor, img_true: torch.Tensor):
    img_pred_np = np.array(img_pred)#.squeeze())
    img_true_np = np.array(img_true)#.squeeze())
    padded0, padded1 = img_pred_np, img_true_np

    hist, bin_edges = np.histogramdd(
        [np.reshape(padded0, -1), np.reshape(padded1, -1)],
        density=True,
    )

    H0 = entropy(np.sum(hist, axis=0))
    H1 = entropy(np.sum(hist, axis=1))
    H01 = entropy(np.reshape(hist, -1))

    return H0 + H1 - H01


# def fsim(img_pred: torch.Tensor, img_true: torch.Tensor):
#     print(img_pred.shape)
#     return piq.fsim(img_pred.unsqueeze(0).unsqueeze(0), img_true.unsqueeze(0).unsqueeze(0))


# def fsim(img_pred: torch.Tensor, img_true: torch.Tensor):
#     img_pred_np = np.array(img_pred.squeeze())
#     img_true_np = np.array(img_true.squeeze())
#     print(img_pred.shape)
#     return quality_metrics.fsim(img_true_np, img_pred_np)
#     # return piq.fsim(img_pred.unsqueeze(0).unsqueeze(0), img_true.unsqueeze(0).unsqueeze(0))


def _gradient_magnitude(img: np.ndarray, img_depth):
    """
    Calculate gradient magnitude based on Scharr operator
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)


def _similarity_measure(x, y, constant):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator


def fsim(img_pred: torch.Tensor, img_true: torch.Tensor, T1=0.85, T2=160) -> float:
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM)
    There are different ways to implement PC, the authors of the original FSIM paper use the method
    defined by Kovesi (1999). The Python phasepack project fortunately provides an implementation
    of the approach.
    There are also alternatives to implement GM, the FSIM authors suggest to use the Scharr
    operation which is implemented in OpenCV.
    Note that FSIM is defined in the original papers for grayscale as well as for RGB images. Our use cases
    are mostly multi-band images e.g. RGB + NIR. To accommodate for this fact, we compute FSIM for each individual
    band and then take the average.
    Note also that T1 and T2 are constants depending on the dynamic range of PC/GM values. In theory this parameters
    would benefit from fine-tuning based on the used data, we use the values found in the original paper as defaults.
    Args:
        org_img -- numpy array containing the original image
        pred_img -- predicted image
        T1 -- constant based on the dynamic range of PC values
        T2 -- constant based on the dynamic range of GM values
    """

    alpha = beta = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    pred_img = np.array(img_pred.squeeze())
    org_img = np.array(img_true.squeeze())
    for it in range(1):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(org_img[:, :], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        pc2_2dim = pc(pred_img[:, :], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros((pred_img.shape[0], pred_img.shape[1]), dtype=np.float64)
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)


def en(img: torch.Tensor):
    entropy = skm.shannon_entropy(img)
    return entropy