##Datasetの他にリスト生成したりするとこ
### 10/1多分完成
import glob
import os.path as osp
import random
import numpy as np
import json
import re
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import models, transforms
from Transform import *
class DistancePredictor():
	def __init__(self,class_index):
		self.class_index = class_index

	def get_argmax(self,out):
		maxid = np.argmax(out.detach().numpy())
		predicter_label_name = self.class_index[str(maxid)][0]
		return predicter_label_name

def make_datapath_list():
	rootpath = "../images/"
	target_path = osp.join(rootpath + '*/*.jpg')
	val_list = []
	train_list = []
	for path in glob.glob(target_path):
		if(random.random() < 0.15):
			val_list.append(path)
		else:
			train_list.append(path)
	return val_list, train_list

class Dataset():
	def __init__(self, file_list, transform=None, phase='train'):
		self.file_list = file_list  # ファイルパスのリスト
		self.transform = transform  # 前処理クラスのインスタンス
		self.phase = phase  # train or valの指定

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		img_path = self.file_list[index]
		img = Image.open(img_path)
		#中の計算部分は貼る位置のピクセル調整。あまり気にしなくて良い
		img_transformed = self.transform(img, self.phase)  # torch.Size([3, 224, 224])
		label = img_path.split("/")
		#label = img_path[3:18]
		#label = re.findall('.*?(\d+).*',label)
		label = label[2]
		#print(label)
		label = self.make_label(label)
		
		return img_transformed, label

	def make_label(self,label):
		if label == "aquarius":
			label = 0
		if label == "ayataka_brown_rice":
			label = 1
		if label == "calpis":
			label = 2
		if label == "craft_boss_black":
			label = 3
		if label == "craft_boss_latte":
			label = 4
		if label == "crystal_geyser":
			label = 5
		if label == "fresh_tea":
			label = 6
		if label == "green_dakara":
			label = 7
		if label == "irohas":
			label = 8
		if label == "sprite":
			label = 9
		if label == "tropicana":
			label = 10
		if label == "wilkinson":
			label = 11
		return label	

if __name__ == "__main__":
	batch_size = 32
	size = 224
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	classes = 4

	val_list,train_list = make_datapath_list()

	train_dataset = Dataset(
		file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')

	val_dataset = Dataset(
		file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')
	index = 1
	print(train_dataset.__getitem__(index)[0].size())
	print(train_dataset.__getitem__(index)[1])

# DataLoaderを作成
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset, batch_size=batch_size, shuffle=True)

	val_dataloader = torch.utils.data.DataLoader(
		val_dataset, batch_size=batch_size, shuffle=False)

# 辞書型変数にまとめる
	dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作確認
	batch_iterator = iter(dataloaders_dict["train"])  # イテレータに変換
	inputs, labels = next(batch_iterator)
	inputs, labels = next(batch_iterator)
	print(inputs.size()) #batch_size,color,height,weight
	print(labels)

	pd.DataFrame(val_list).to_csv("data/val.csv",index=False,header=False)
	pd.DataFrame(train_list).to_csv("data/train.csv",index=False,header=False)
	f = pd.read_csv("data/train.csv")
