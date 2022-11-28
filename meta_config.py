# configuration file for training
# Author: Simon Zhou, last modify Nov.18 2022

'''
Do not need a change log, you can always change to your own directory
'''
import os
data_dir = os.getcwd()
res_dir = os.getcwd() + "/res"

test_data_dir = os.getcwd() + "/testset"

test_num = 20
train_val_ratio = 0.8

# lambdas in loss function
lambda1 = 1
lambda2 = 1

# vgg block for perceptual loss
block_idx = [0,1,2]