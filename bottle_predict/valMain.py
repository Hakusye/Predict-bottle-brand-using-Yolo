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

def ShowArea(predicted,val_labels,batch_size):
	for k in range(batch_size):
		for i in range(12):
			for j in range(12):
				locate = i*12+j
				if(locate == predicted[k] and locate == val_labels[k]):
					print("o ",end="")
				elif(locate == predicted[k]):
					print("p ",end="")
				elif(locate == val_labels[k]):
					print("v ",end="")
				else:
					print("x ",end="")
			print()
		print("==================")
	print("========batch==========")

def missDistance(val_label,predicted,batch_size,di):
	for i in range(len(val_label)):
		MissDistance = (int(val_label[i]%12) - int(predicted[i]%12))**2 + (int(val_label[i]/12) - int(predicted[i]/12))**2
		if(MissDistance<40):
			di[MissDistance]+=1
		else:
			di[40]+=1
	return di

def DiResult(di):
	for i in range(len(di)):
		print("Distance," + str(i) + "," + str(di[i]))

def Accuracy(model,val_data_loder,device,batch_size):
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
		#ShowArea(predicted,val_labels,batch_size)
		#加算式
		#di = missDistance(val_labels,predicted,batch_size,di)
	#DiResult(di)
	print('正解率：%d / %d = %f'% (count_when_correct, total, int(count_when_correct)/int(total)))
	model.train()

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)
	batch_size = 16
	size = 224
	Stime = 0.0
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	classes = 12
	val_list = []

	alr_train_path = "weights/ResNet50_batch32_epoch19.pth"
	#class_index = json.load(open(ClassIndex.json))
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
