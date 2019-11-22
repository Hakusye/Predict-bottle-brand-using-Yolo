import csv
import glob
import os.path as osp
import random
import numpy as np
import json
import time
from PIL import Image
import matplotlib.pyplot as plt
#%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import models, transforms
import torch.optim as optimizer
from MakeDataset import *
#from Model import *
from Transform import *

def Accuracy(model,val_data_loader,device,batch_size):
	di = [0 for i in range(41)]
	model.to(device)
	model.eval()
	count_when_correct = 0
	total = 0
	for data in val_data_loader:
		val_data, val_labels = data
		val_data,val_labels = val_data.to(device),val_labels.to(device)
		results = model(val_data).to(device)
		_, predicted = torch.max(results.data, 1)
		total += val_labels.size(0)
		count_when_correct += (predicted == val_labels).sum()
	print('正解率：%d / %d = %f'% (count_when_correct, total, int(count_when_correct)/int(total)))
	model.train()

if __name__ == "__main__":
	config = configurations["config"]
	class_name = configurations["class_name"]
	rev_class_name = configurations["rev_class_name"]
	device = config["DEVICE"]
	batch_size = config["BATCH_SIZE"]
	size = config["IMAGE_SIZE"]
	mean = config["MEAN"]
	std = config["STD"]
	classes = config["CLASSES"]
	val_list = []
	alr_train_path = "weights/ResNet50_batch32_epoch19.pth"
	
	model = models.resnet50(pretrained=False)
	model.fc = nn.Linear(2048,classes)
	model.load_state_dict(torch.load(alr_train_path))
	#val_list,train_list = make_datapath_list()

	with open("data/val.csv", "r") as f:
		val_list = [s.strip() for s in f.readlines()]

	val_dataset = Dataset(
		file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')
	
	val_data_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=batch_size, shuffle=False)
#dataloder作成
###変更対象
	Accuracy(model,val_data_loader,device,batch_size)
