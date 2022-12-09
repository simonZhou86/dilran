# inference fused image

import os
import argparse
import torch
import torch.nn as nn
import sys
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from models.model_v5 import *
from our_utils import *
from eval import psnr, ssim, mutual_information
from evaluation_metrics import fsim, nmi, en
import skimage.io as io
# import sys
# sys.path.append("./model")

parser = argparse.ArgumentParser(description='Inference Fused Image configs')
parser.add_argument('--test_folder', type=str, default='./testset', help='input test image')
parser.add_argument('--model', type=str, default='./res/pretrained_models/model_v5/last.pt', help='which model to use')
parser.add_argument('--save_folder', type=str, default='./res/fused_image', help='input image to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda', default='true')

opt = parser.parse_args()

########### gpu ###############
device = torch.device("cuda:0" if opt.cuda else "cpu")
###############################

######### make dirs ############
# save_dir = os.path.join(opt.save_folder, "model_vf_sfnnMean")
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
###############################

####### loading pretrained model ########
model = fullModel().to(device)
model.load_state_dict(torch.load(opt.model))
#model = torch.load(opt.model)
#########################################

########### loading test set ###########
test_ct = torch.load(os.path.join(opt.test_folder, 'ct_test.pt')).to(device)
test_mri = torch.load(os.path.join(opt.test_folder, 'mri_test.pt')).to(device)
########################################

psnr = PeakSignalNoiseRatio()
psnrs = []
ssims = []
nmis = []
mis = []
fsims = []
ens = []
for slice in range(test_ct.shape[0]):
    # if slice > 0:
    #     break
    ct_slice = test_ct[slice,:,:,:].unsqueeze(0)
    mri_slice = test_mri[slice,:,:,:].unsqueeze(0)

    ct_fe = model.fe(ct_slice)
    #print(ct_fe.shape)
    mri_fe = model.fe(mri_slice)

    fused = fusion_strategy(ct_fe, mri_fe, device, "SFNN")
    #fused = torch.maximum(ct_fe, mri_fe)
    final = model.recon(fused)
    #print(final.squeeze(0).squeeze(0))

    final = final.squeeze(0).squeeze(0).detach().cpu().clamp(min=0, max=1)
    gt1 = ct_slice.squeeze(0).squeeze(0).cpu().clamp(min=0, max=1)
    #print(torch.min(gt1), torch.max(gt1))
    gt2 = mri_slice.squeeze(0).squeeze(0).cpu().clamp(min=0, max=1)

    # io.imsave(os.path.join(save_dir, "fused_{}.jpg".format(slice)), (final.numpy() * 255).astype(np.uint8))
    # io.imsave(os.path.join(save_dir, "mri_{}.jpg".format(slice)), (gt2.numpy() * 255).astype(np.uint8))
    # io.imsave(os.path.join(save_dir, "ct_{}.jpg".format(slice)), (gt1.numpy() * 255).astype(np.uint8))

    #print("image {} saved".format(slice))

    psnr_val1 = psnr(final, gt1)
    psnr_val2 = psnr(final, gt2)
    psnr_val = (psnr_val1 + psnr_val2) / 2
    #print(psnr_val.item())
    psnrs.append(psnr_val.item())

    ssim_val1 = ssim(final, gt1)
    ssim_val2 = ssim(final, gt2)
    ssim_val = (ssim_val1 + ssim_val2) / 2
    #print(ssim_val)
    ssims.append(ssim_val)

    nmi_val1 = nmi(final, gt1)
    nmi_val2 = nmi(final, gt2)
    nmi_val = (nmi_val1 + nmi_val2) / 2
    #print(nmi_val)
    nmis.append(nmi_val)

    mi_val1 = mutual_information(final, gt1)
    mi_val2 = mutual_information(final, gt2)
    mi_val = (mi_val1 + mi_val2) / 2
    #print(mi_val)
    mis.append(mi_val)

    fsim_val1 = fsim(final, gt1)
    fsim_val2 = fsim(final, gt2)
    fsim_val = (fsim_val1 + fsim_val2) / 2
    fsims.append(fsim_val)

    en_val = en(final)
    ens.append(en_val)

    #plt.imshow(ct_fe[0,32,:,:].detach().cpu().numpy(), "gray")
    #plt.show()
# plt.figure(figsize=(12, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(ct_slice.squeeze(0).squeeze(0).cpu().numpy(), "gray")
# plt.title("CT Slice")

# plt.subplot(1, 3, 2)
# plt.imshow(mri_slice.squeeze(0).squeeze(0).cpu().numpy(), "gray")
# plt.title("MRI Slice")

# plt.subplot(1, 3, 3)
# plt.imshow(final.numpy(), "gray")
# plt.title("Fused Slice")

# plt.show()

print("psnrs")
print(sum(psnrs) / len(psnrs))
print("ssims")
print(sum(ssims) / len(ssims))
print("nmis")
print(sum(nmis) / len(nmis))
print("mis")
print(sum(mis) / len(mis))
print("fsims")
print(sum(fsims) / len(fsims))
print("entropy")
print(sum(ens) / len(ens))