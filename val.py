# Validation script for the project
# Validate a trained medical image fusion model
# Author: Reacher, last modify Nov. 28, 2022

'''
Change log:
Reacher: file created
'''

from evaluation_metrics import *

# run validation for every epoch


import os
import argparse
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import *
from our_utils import *

test_folder = './testset'
save_folder = './res/fused_image'
output_filename = None
cuda = True

########### gpu ###############
device = torch.device("cuda:0" if cuda else "cpu")
###############################

######### make dirs ############
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
###############################

####### loading pretrained model ########

#########################################

########### loading test set ###########
test_ct = torch.load(os.path.join(test_folder, 'ct_test.pt')).to(device)
test_mri = torch.load(os.path.join(test_folder, 'mri_test.pt')).to(device)


########################################

# psnr = PeakSignalNoiseRatio()

# for strategy in [ "addition", "average", "FER", "L1NW", "AL1NW", "FL1N"]:
# for strategy in ["average", "max_val", "FER", "FL1N"]:


def validate(model_pt):
    model = fullModel().to(device)
    model.load_state_dict(torch.load(model_pt, map_location=device))
    # model.eval()
    # Use SFNN strategy
    for strategy in ["SFNN"]:
        psnrs, ssims, nmis, mis, fsims = [], [], [], [], []
        for slice in range(test_ct.shape[0]):
            ct_slice = test_ct[slice, :, :, :].unsqueeze(0)
            mri_slice = test_mri[slice, :, :, :].unsqueeze(0)

            ct_fe = model.fe(ct_slice)
            # print(ct_fe.shape)
            mri_fe = model.fe(mri_slice)

            fused = fusion_strategy(ct_fe, mri_fe, device=device, strategy=strategy)
            final = model.recon(fused)
            final = final.squeeze(0).squeeze(0).detach().cpu().clamp(min=0, max=1.)
            gt1 = ct_slice.squeeze(0).squeeze(0).cpu().clamp(min=0, max=1.)
            gt2 = mri_slice.squeeze(0).squeeze(0).cpu().clamp(min=0, max=1.)

            psnr_val1 = psnr(final, gt1)
            psnr_val2 = psnr(final, gt2)
            psnr_val = (psnr_val1 + psnr_val2) / 2
            psnrs.append(psnr_val)

            ssim_val1 = ssim(final.unsqueeze(0).unsqueeze(0), gt1.unsqueeze(0).unsqueeze(0))
            ssim_val2 = ssim(final.unsqueeze(0).unsqueeze(0), gt2.unsqueeze(0).unsqueeze(0))
            ssim_val = (ssim_val1 + ssim_val2) / 2
            ssims.append(ssim_val)

            nmi_val1 = nmi(final, gt1)
            nmi_val2 = nmi(final, gt2)
            nmi_val = (nmi_val1 + nmi_val2) / 2
            nmis.append(nmi_val)

            mi_val1 = mutual_information(final, gt1)
            mi_val2 = mutual_information(final, gt2)
            mi_val = (mi_val1 + mi_val2) / 2
            mis.append(mi_val)

            fsim_val1 = fsim(final, gt1)
            fsim_val2 = fsim(final, gt2)
            fsim_val = (fsim_val1 + fsim_val2) / 2
            fsims.append(fsim_val)

        # print(len(psnrs))
        print(strategy)
        # print(f"Average PSNR: {np.mean(psnrs)}")
        # print(f"Average SSIM: {np.mean(ssims)}")
        # print(f"Average NMI: {np.mean(nmis)}")
        # print(f"Average MI: {np.mean(mis)}")
        # print("---------------------")
        val_psnr = np.mean(psnrs)
        val_ssim = np.mean(ssims)
        val_nmi = np.mean(nmis)
        val_mi = np.mean(mis)
        val_fsim = np.mean(fsims)
        return val_psnr, val_ssim, val_nmi, val_mi, val_fsim
