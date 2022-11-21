# Training script for the project
# Author: Simon Zhou, last modify Nov. 18, 2022

'''
Change log:
-Simon: file created, write some training code

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

import meta_config as c
from model import *
from our_utils import *
from dataset_loader import *
from loss import *

import wandb


parser = argparse.ArgumentParser(description='parameters for the training script')
parser.add_argument('--dataset', type=str, default="CT-MRI", help="which dataset to use, available option: CT-MRI, MRI-PET, MRI-SPECT")
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')
parser.add_argument('--lr_decay', type=bool, default=False, help='decay learing rate?')
#parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
parser.add_argument('--cuda', action='store true', help='whether to use cuda', default= True)
parser.add_argument('-seed', type=int, default=3407, help='random seed to use')
opt = parser.parse_args()

######### whether to use cuda ####################
device = torch.device("cuda:0" if opt.cuda else "cpu")
#################################################

## seeding ####################
seed_val = opt.seed
random_seed(seed_val, opt.cuda)
################################

####### loading dataset ####################################
target_dir = os.path.join(c.data_dir, opt.dataset)
ct, mri = get_common_file(target_dir)
train_ct, train_mri, test_ct, test_mri = load_data(ct, target_dir, c.test_num)
torch.save(test_ct, os.path.join(c.test_data_dir, "ct_test.pt"))
torch.save(test_mri, os.path.join(c.test_data_dir, "mri_test.pt"))
#print(train_ct.shape, train_mri.shape, test_ct.shape, test_mri.shape)

train_total = torch.cat((train_ct, train_mri), dim = 0)

# these loaders return index, not the actual image 
train_loader, val_loader = get_loader(train_ct, train_mri, c.train_val_ratio, opt.batch_size)
print("train loader length: ", len(train_loader), " val loder length: ", len(val_loader))
############################################################

############ making dirs########################
model_dir = os.path.join(c.res_dir), "pretrained_models"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
################################################

############ loading model #####################
model = fullModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

if opt.lr_decay:
    stepLR = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.5)
###################################################

############## train model ##############
wandb.init(project="test-project", entity="csc2529") # visualize in wandb

wandb.config = {
  "learning_rate": opt.lr,
  "epochs": opt.epochs,
  "batch_size": opt.batch_size
}

train_loss = []
val_loss = []
t = trange(opt.epochs, desc='Training progress...', leave=True)
lowest_val_loss = int(1e9)

for i in range(t):
    print("new epoch {} starts!".format(i))
    # clear gradient in model
    model.zero_grad()
    loss = 0
    # train model
    model.train()
    for j, batch_inx in enumerate(train_loader):
        # clear gradient in optimizer
        optimizer.zero_grad()
        batch_idx = batch_idx.view(-1).long()
        img = train_total[batch_idx]
        img_out = model(img)
        # compute loss
        loss += loss_func(img_out, img, c.lambda1, c.lambda2, c.block_idx, device)
        # back propagate and update weights
        loss.backward()
        optimizer.step()
    # lr decay
    if opt.lr_decay:
        stepLR.step()
    # store loss
    ave_loss = loss.item() / len(train_loader)
    train_loss.append(ave_loss)
    print("epoch {}, training loss is: {}".format(i), ave_loss)

    # validation
    val_loss = []
    with torch.no_grad():
        loss = 0
        # eval model, unable update weights
        model.eval()
        for k, batch_idx in enumerate(val_loader):
            batch_idx = batch_idx.view(-1).long()
            val_img = train_total[batch_idx]
            val_img_out = model()
            loss += loss_func(img_out, img, c.lambda1, c.lambda2, c.block_idx, device)

    ave_val_loss = loss.item() / len(val_loader)
    val_loss.append(ave_val_loss)
    print("epoch {}, validation loss is: {}".format(i), ave_val_loss)

    # save model
    if val_loss < lowest_val_loss:
        torch.save(model.state_dict(), model_dir)
        lowest_val_loss = val_loss
    print("model is saved in epoch {}".format(i))
########################################