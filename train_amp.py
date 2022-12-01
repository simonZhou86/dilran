# training script using torch amp
# torch amp can only be used when cuda=True

import argparse
import os
import sys
sys.path.append("../")
from tqdm import trange

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import vgg16_bn
# print(torch.__version__)
# print(torchvision.__version__)

import meta_config as c
from model_baseline import *
from our_utils import *
from dataset_loader import *
from loss import *

import wandb

#sys.exit()
parser = argparse.ArgumentParser(description='parameters for the training script')
parser.add_argument('--dataset', type=str, default="CT-MRI", help="which dataset to use, available option: CT-MRI, MRI-PET, MRI-SPECT")
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training')
parser.add_argument('--lr_decay', type=bool, default=False, help='decay learing rate?')
parser.add_argument('--accum_batch', type=int, default=1, help='number of batches for gradient accumulation')
parser.add_argument('--lambda1', type=float, default=0.5, help='weight for image gradient loss')
parser.add_argument('--lambda2', type=float, default=0.5, help='weight for perceptual loss')
#parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda', default= True)
parser.add_argument('--seed', type=int, default=3407, help='random seed to use')
parser.add_argument('--base_loss', type=str, default='l2_norm', help='which loss function to use for pixel-level (l2 or l1 charbonnier)')
opt = parser.parse_args()

######### whether to use cuda ####################
device = torch.device("cuda:0" if opt.cuda else "cpu")
#################################################

USE_AMP = True # use automatic mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

########## seeding ##############
seed_val = opt.seed
random_seed(seed_val, opt.cuda)
################################

############ making dirs########################
if not os.path.exists(c.res_dir):
    os.mkdir(c.res_dir)
model_dir = os.path.join(c.res_dir, "pretrained_models")
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(c.test_data_dir):
    os.mkdir(c.test_data_dir)
################################################

####### loading dataset ####################################
target_dir = os.path.join(c.data_dir, opt.dataset)
ct, mri = get_common_file(target_dir)
train_ct, train_mri, test_ct, test_mri = load_data(ct, target_dir, c.test_num)
# torch.save(test_ct, os.path.join(c.test_data_dir, "ct_test.pt"))
# torch.save(test_mri, os.path.join(c.test_data_dir, "mri_test.pt"))
#print(train_ct.shape, train_mri.shape, test_ct.shape, test_mri.shape)

train_total = torch.cat((train_ct, train_mri), dim = 0).to(device)

# these loaders return index, not the actual image
train_loader, val_loader = get_loader(train_ct, train_mri, c.train_val_ratio, opt.batch_size)
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

############ loading model #####################
model = fullModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

if opt.lr_decay:
    stepLR = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.5)
###################################################

##### downloading pretrained vgg model ##################
vgg = vgg16_bn(pretrained = True)
########################################################

############## train model ##############
wandb.init(project="test-project", entity="csc2529", config=opt) # visualize in wandb

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
        #img = train_total[batch_idx]

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            img = train_total[batch_idx]
            img_out = model(img)
            # compute loss
            loss,_,_,_ = loss_func2(vgg, img_out, img, opt.lambda1, opt.lambda2, c.block_idx, device)
            # back propagate and update weights
            #print("batch reg, grad, percep loss: ", reg_loss.item(), img_grad.item(), percep.item())
            #loss = loss / NUM_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        #loss.backward()

        #if ((j + 1) % NUM_ACCUMULATION_STEPS == 0) or (j + 1 == len(train_loader)):
        #optimizer.step()
        scaler.step(optimizer)
        
        b_loss += loss.item()
        #wandb.log({"loss": loss})

        scaler.update()
    
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
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                val_img = train_total[batch_idx]
                val_img_out = model(val_img)
                # display first image to visualize, this can be changed
                val_display_img.extend([val_img_out[i].squeeze(0).cpu().numpy() for i in range(1)])
                loss, _,_,_= loss_func2(vgg, img_out, img, opt.lambda1, opt.lambda2, c.block_idx, device)
            b_loss += loss.item()

    ave_val_loss = b_loss / len(val_loader)
    val_loss.append(ave_val_loss)
    print("epoch {}, validation loss is: {}".format(i, ave_val_loss))

    wandb.log({"train loss": ave_loss, "epoch": i})
    wandb.log({"val loss": ave_val_loss, "epoch": i})
    wandb.log({"val sample images": [wandb.Image(img) for img in val_display_img]})

    # save model
    if ave_val_loss < lowest_val_loss:
        torch.save(model.state_dict(), model_dir+"/model_at_{}.pt".format(i))
        lowest_val_loss = ave_val_loss
        print("model is saved in epoch {}".format(i))
    
    # lr decay update
    if opt.lr_decay:
        stepLR.step()
########################################


class grad_loss(nn.Module):
    '''
    image gradient loss
    '''
    def __init__(self, device, amp = True, vis = False, type = "sobel"):

        super(grad_loss, self).__init__()
        
        # only use sobel filter now
        if type == "sobel":
            #with torch.cuda.amp.autocast(enabled=amp):
            kernel_x = torch.HalfTensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
            kernel_y = torch.HalfTensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])

        kernel_x = kernel_x.unsqueeze(0).unsqueeze(0)
        kernel_y = kernel_y.unsqueeze(0).unsqueeze(0)
        # do not want update these weights
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).to(device)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).to(device)
        #self.weight_yx = nn.Parameter(data=kernel_x.double(), requires_grad=False).to(device)
        #self.weight_yy = nn.Parameter(data=kernel_y.double(), requires_grad=False).to(device)
        self.vis = vis
    
    def forward(self, x, y):
        # conv2d to find image gradient in x direction and y direction
        # of input image x and image y
        grad_xx = F.conv2d(x, self.weight_x)
        grad_xy = F.conv2d(x, self.weight_y)
        grad_yx = F.conv2d(y, self.weight_x)
        grad_yy = F.conv2d(y, self.weight_y)

        if self.vis:
            return grad_xx, grad_xy, grad_yx, grad_yy
        
        # total image gradient, in dx and dy direction for image X and Y
        # gradientX = torch.abs(grad_xx) + torch.abs(grad_xy)
        # gradientY = torch.abs(grad_yx) + torch.abs(grad_yy)
        x_diff = ((torch.abs(grad_xx) - torch.abs(grad_yx)) ** 2).mean()
        y_diff = ((torch.abs(grad_xy) - torch.abs(grad_yy)) ** 2).mean()
        
        # mean squared frobenius norm (||.||_F^2)
        #grad_f_loss = torch.mean(torch.pow(torch.norm((gradientX - gradientY), p = "fro"), 2))
        grad_f_loss = x_diff + y_diff
        return grad_f_loss