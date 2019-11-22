import glob
import csv
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
from valMain import *
from config import configurations

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = configurations["config"]
class_name = configurations["class_name"]
rev_class_name = configurations["rev_class_name"]
device = config["DEVICE"]
batch_size = config["BATCH_SIZE"]
size = config["IMAGE_SIZE"]
mean = config["MEAN"]
std = config["STD"]
classes = config["CLASSES"]
Stime = 0.0
#val_list,train_list = make_datapath_list()
train_list = []
val_list = []
with open("data/train.csv","r") as f:
	train_list = [s.strip() for s in f.readlines()]

with open("data/val.csv","r") as f:
	val_list = [s.strip() for s in f.readlines()]

train_dataset = Dataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = Dataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')
print(len(train_dataset))
#dataloder作成
train_data_loader = torch.utils.data.DataLoader(
	train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(
	val_dataset, batch_size=batch_size, shuffle=False)
#辞書型変数にまとめる
dataloaders_dict = {"train": train_data_loader, "val": val_data_loader}

#ファインチューニング準備
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048,classes)
#alr_train_path = "weights/ResNet50_batch64_epoch50.pth"

criterion = nn.CrossEntropyLoss()
optimizer = optimizer.SGD(model.parameters(),lr=config["WEIGHT_DECAY"])
MAX_EPOCH=config["MAX_EPOCH"]
model.to(device)
model.train()
#torch.backends.cudnn.benchmark = True
#学習
cnt=0
Stime = time.time()
for epoch in range(MAX_EPOCH):
	running_loss = 0.0
	for i,data in enumerate(train_data_loader):
		cnt+=1
# dataから学習対象データと教師ラベルデータ
		train_data, teacher_labels = data
		train_data,teacher_labels = train_data.to(device),teacher_labels.to(device)
		optimizer.zero_grad()
		outputs = model(train_data).to(device)
		loss = criterion(outputs, teacher_labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		#if cnt % 30 == 29:
		#定数のとこをtrain.sizeとかでまとめたい
		#	print('['+ str(epoch)+',' + str(i+1) + ']  ' + str(running_loss))
		#	print("time:"+str(time.time()-Stime))
		#	Stime = time.time()
		#	running_loss = 0.
		if(cnt % 30 == 29):
			cnt=0
			Accuracy(model,val_data_loader ,device,batch_size)
			print('savefile:' + 'weights/ResNet50_batch' + str(batch_size) + '_epoch' + str(epoch) + '.pth')
			torch.save(model.state_dict(),'weights/ResNet50_batch'+ str(batch_size) + '_epoch' + str(epoch) + '.pth')

print('savefile:' + 'weights/ResNet50_batch' + str(batch_size) + '_epoch' + str(epoch) + '.pth')
torch.save(model.state_dict(),'weights/ResNet50_batch'+ str(batch_size) + '_epoch' + str(epoch) + '.pth')

print("val")
