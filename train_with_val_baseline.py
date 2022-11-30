# Training script for the project
# Author: Simon Zhou, last modify Nov. 18, 2022

'''
Change log:
-Simon: file created, write some training code
-Simon: refine training script
-Reacher: train v3
-Reacher: add model choice
'''

import argparse
import os
import sys

sys.path.append("../")
from tqdm import trange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16_bn

import meta_config as config
from model import *
from our_utils import *
from dataset_loader import *
from loss import *
from val_baseline import validate
from model_msrpan import SRN

import wandb

parser = argparse.ArgumentParser(description='parameters for the training script')
parser.add_argument('--model', type=str, default="ours",
                    help="which model to use, available option: ours, MSRPAN")
parser.add_argument('--dataset', type=str, default="CT-MRI",
                    help="which dataset to use, available option: CT-MRI, MRI-PET, MRI-SPECT")
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training')
parser.add_argument('--lr_decay', type=bool, default=False, help='decay learing rate?')
parser.add_argument('--accum_batch', type=int, default=1, help='number of batches for gradient accumulation')
parser.add_argument('--lambda1', type=float, default=0.5, help='weight for image gradient loss')
parser.add_argument('--lambda2', type=float, default=0.5, help='weight for perceptual loss')
# parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda', default=True)
parser.add_argument('--seed', type=int, default=3407, help='random seed to use')
parser.add_argument('--base_loss', type=str, default='l1_charbonnier',
                    help='which loss function to use for pixel-level (l2 or l1 charbonnier)')

opt = parser.parse_args()

######### whether to use cuda ####################
device = torch.device("cuda:0" if opt.cuda else "cpu")
#################################################

########## seeding ##############
seed_val = opt.seed
random_seed(seed_val, opt.cuda)
################################

############ making dirs########################
if not os.path.exists(config.res_dir):
    os.mkdir(config.res_dir)
model_dir = os.path.join(config.res_dir, "pretrained_models")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(config.test_data_dir):
    os.mkdir(config.test_data_dir)
################################################

####### loading dataset ####################################
target_dir = os.path.join(config.data_dir, opt.dataset)
ct, mri = get_common_file(target_dir)
train_ct, train_mri, test_ct, test_mri = load_data(ct, target_dir, config.test_num)
torch.save(test_ct, os.path.join(config.test_data_dir, "ct_test.pt"))
torch.save(test_mri, os.path.join(config.test_data_dir, "mri_test.pt"))
# print(train_ct.shape, train_mri.shape, test_ct.shape, test_mri.shape)

train_total = torch.cat((train_ct, train_mri), dim=0).to(device)

# these loaders return index, not the actual image
train_loader, val_loader = get_loader(train_ct, train_mri, config.train_val_ratio, opt.batch_size)
print("train loader length: ", len(train_loader), " val loder length: ", len(val_loader))

# check the seed is working
# for batch_idx in train_loader:
#     batch_idx = batch_idx.view(-1).long()
#     print(batch_idx)
# print("validation index")
# for batch_idx in val_loader:
#     batch_idx = batch_idx.view(-1).long()
#     print(batch_idx)
# sys.exit()
############################################################


"""
 choose model
"""
if opt.model == "ours":
    model = fullModel().to(device)
    print("Training ours")
elif opt.model == "MSRPAN":
    model = SRN().to(device)
    print("Training MSRPAN")
# default
else:
    model = fullModel().to(device)
    print("Default: Training ours")


############ loading model #####################
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

if opt.lr_decay:
    stepLR = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
###################################################

##### downloading pretrained vgg model ##################
vgg = vgg16_bn(pretrained=True)
########################################################

############## train model ##############
wandb.init(project="test-project", entity="csc2529", config=opt)  # visualize in wandb

# wandb.config = {
#   "learning_rate": opt.lr,
#   "epochs": opt.epochs,
#   "batch_size": opt.batch_size,
#   "lambda1": c.lambda1,
#   "lambda2": c.lambda2
# }

wandb.watch(model)

# gradient accumulation for small batch
NUM_ACCUMULATION_STEPS = opt.accum_batch

train_loss = []
val_loss = []
t = trange(opt.epochs, desc='Training progress...', leave=True)
lowest_val_loss = int(1e9)
best_ssim = 0

for i in t:
    print("new epoch {} starts!".format(i))
    # clear gradient in model
    model.zero_grad()
    b_loss = 0
    # train model
    model.train()
    for j, batch_idx in enumerate(train_loader):
        # clear gradient in optimizer
        optimizer.zero_grad()
        batch_idx = batch_idx.view(-1).long()
        img = train_total[batch_idx]
        img_out = model(img)
        # compute loss
        loss, _, _, _ = loss_func2(vgg, img_out, img, opt.lambda1, opt.lambda2, config.block_idx, device)
        # back propagate and update weights
        # print("batch reg, grad, percep loss: ", reg_loss.item(), img_grad.item(), percep.item())
        # loss = loss / NUM_ACCUMULATION_STEPS
        loss.backward()

        # if ((j + 1) % NUM_ACCUMULATION_STEPS == 0) or (j + 1 == len(train_loader)):
        optimizer.step()
        b_loss += loss.item()
        # wandb.log({"loss": loss})

    # store loss
    ave_loss = b_loss / len(train_loader)
    train_loss.append(ave_loss)
    print("epoch {}, training loss is: {}".format(i, ave_loss))

    # validation
    val_loss = []
    val_display_img = []
    with torch.no_grad():
        b_loss = 0
        # eval model, unable update weights
        model.eval()
        for k, batch_idx in enumerate(val_loader):
            batch_idx = batch_idx.view(-1).long()
            val_img = train_total[batch_idx]
            val_img_out = model(val_img)
            # display first image to visualize, this can be changed
            val_display_img.extend([val_img_out[i].squeeze(0).cpu().numpy() for i in range(1)])
            loss, _, _, _ = loss_func2(vgg, img_out, img, opt.lambda1, opt.lambda2, config.block_idx, device)
            b_loss += loss.item()

    ave_val_loss = b_loss / len(val_loader)
    val_loss.append(ave_val_loss)
    print("epoch {}, validation loss is: {}".format(i, ave_val_loss))

    # define a metric we are interested in the minimum of
    wandb.define_metric("train loss", summary="min")
    # define a metric we are interested in the maximum of
    wandb.define_metric("val loss", summary="min")

    wandb.log({"train loss": ave_loss, "epoch": i})
    wandb.log({"val loss": ave_val_loss, "epoch": i})
    wandb.log({"val sample images": [wandb.Image(img) for img in val_display_img]})

    # save model
    if ave_val_loss < lowest_val_loss:
        torch.save(model.state_dict(), model_dir + "/model_at_{}.pt".format(i))
        lowest_val_loss = ave_val_loss
        print("model is saved in epoch {}".format(i))

    # Evaluate during training
    # Save the current model
    torch.save(model.state_dict(), model_dir + "/current.pt".format(i))

    val_psnr, val_ssim, val_nmi, val_mi, val_fsim = validate(model_dir + "/current.pt")

    # define a metric we are interested in the maximum of
    wandb.define_metric("PSNR", summary="max")
    wandb.define_metric("SSIM", summary="max")
    wandb.define_metric("NMI", summary="max")
    wandb.define_metric("MI", summary="max")
    wandb.define_metric("FSIM", summary="max")

    wandb.log({"PSNR": val_psnr, "epoch": i})
    wandb.log({"SSIM": val_ssim, "epoch": i})
    wandb.log({"NMI": val_nmi, "epoch": i})
    wandb.log({"MI": val_mi, "epoch": i})
    wandb.log({"FSIM": val_fsim, "epoch": i})

    print("PSNR", "SSIM", "NMI", "MI", "FSIM")
    print(val_psnr, val_ssim, val_nmi, val_mi, val_fsim)
    if val_ssim > best_ssim:
        best_ssim = val_ssim
        print(f"ヾ(◍°∇°◍)ﾉﾞ New best SSIM = {best_ssim}")
        # overwrite
        torch.save(model.state_dict(), model_dir + "/best.pt".format(i))

    if i == opt.epochs - 1:
        torch.save(model.state_dict(), model_dir + "/last.pt".format(i))

    # lr decay update
    if opt.lr_decay:
        stepLR.step()
########################################
