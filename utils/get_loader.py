'''
This script aims to do the following things:
1. one loader for domain A: ProstateX
2. one loader for domain B: KGH

Last Modify: 2021 12.28 by Simon Zhou
load resized data 
'''
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import sys
sys.path.append("../")
from utils.normalize import normalize
import os
import math


class getPatient(Dataset):
	def __init__(self, total_len):
		self.tmp = total_len
		self.total_len = self.tmp
	
	def __len__(self):
		return self.total_len
	
	def __getitem__(self, ind):
		return torch.Tensor([ind])


def loader(data, batch_s):
	'''
	This function returns a loader for prostateX or KGH images

	data: tensor for prostateX images or KGH images
	batch_s: batch size (set to 2)
	'''
	#img = torch.load(data)
	# for i in range(data.shape[0]):
	# 	normalize(data[i])
	
	total_len = data.shape[0]
	#n_train = total_len
	#train_set, val_set = random_split(img, lengths=(n_train, total_len-n_train))
	trainB_ind = getPatient(total_len)
	train_loader = DataLoader(trainB_ind, batch_size=batch_s, num_workers=0, shuffle=False, drop_last=True)
	#val_loader = DataLoader(val_set, batch_size=batch_s, num_workers=0, shuffle=True, drop_last=True)
	return train_loader #val_loader

