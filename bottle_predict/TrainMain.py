###CNNの交差検証これで合ってる気がしない:
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
cross_classes = config["CROSS_CLASSES"]
Stime = 0.0
#val_list,train_list = make_datapath_list()
train_list = []
val_list = []
class_list=[[]for i in range(cross_classes)]
for i in range(cross_classes):
	with open("data/" + str(i) + ".csv","r") as f:
		class_list[i] = [s.strip() for s in f.readlines()]
'''
with open("data/train.csv","r") as f:
	train_list = [s.strip() for s in f.readlines()]

with open("data/val.csv","r") as f:
	val_list = [s.strip() for s in f.readlines()]
'''
class_dataset = []
for i,row in enumerate(class_list):
	class_dataset.append(Dataset(file_list=row, transform=ImageTransform(size, mean, std), phase='train'))
#train_dataset = Dataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
#val_dataset = Dataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')
print(len(class_dataset[0]))
#dataloder作成
class_data_loader1=[]
for dataset in class_dataset:
	class_data_loader1.append(torch.utils.data.DataLoader(
		dataset, batch_size=batch_size, shuffle=True))
class_data_loader2 = class_data_loader1
'''
train_data_loader = torch.utils.data.DataLoader(
	train_dataset, batch_size=batch_size, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(
	val_dataset, batch_size=batch_size, shuffle=False)
'''
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
	
	for val_loader in class_data_loader1:
		for train_loader in class_data_loader2:
			if val_loader == train_loader:
				continue
			for i,data in enumerate(train_loader):
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
		Accuracy(model,val_loader ,device,batch_size)
	print('savefile:' + 'weights/ResNet50_batch' + str(batch_size) + '_epoch' + str(epoch+1) + '.pth')
	torch.save(model.state_dict(),'weights/ResNet50_batch'+ str(batch_size) + '_epoch' + str(epoch+1) + '.pth')

print('savefile:' + 'weights/ResNet50_batch' + str(batch_size) + '_epoch' + str(epoch+1) + '.pth')
torch.save(model.state_dict(),'weights/ResNet50_batch'+ str(batch_size) + '_epoch' + str(epoch+1) + '.pth')

print("fin")
