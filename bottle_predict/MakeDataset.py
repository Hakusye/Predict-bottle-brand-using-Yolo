##Datasetの他にリスト生成したりするとこ
### 任意の数（同確率）のリストを生成することに成功
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
import sys,os 
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from config import configurations
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

def make_datapath_list(classes=2):
	rootpath =  configurations["config"]["ROOT_IMAGES_PATH"]
	rootpath_val =  configurations["config"]["ROOT_IMAGES_PATH_VAL"]
	target_path = osp.join(rootpath + '*/*' + configurations["config"]["EXT"])
	target_path_val = osp.join(rootpath + '*/*' + configurations["config"]["EXT"])
	class_list =[[] for i in range(classes+1)]
	
	for path in glob.glob(target_path):
		class_list[int(random.random()*classes)+1].append(path)

	for path in glob.glob(target_path_val):
		class_list[0].append(path)
	
	target_path = osp.join(rootpath + '*/output/*' + configurations["config"]["EXT"])
	for path in glob.glob(target_path):
		class_list[int(random.random()*classes)+1].append(path)
	
	return class_list

class Dataset():
	def __init__(self, file_list, transform=None, phase='train'):
		self.file_list = file_list  # ファイルパスのリスト
		self.transform = transform  # 前処理クラスのインスタンス
		self.phase = phase  # train or valの指定
		self.config = configurations["config"]
		self.classes = configurations["class_name"]

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		img_path = self.file_list[index]
		img = Image.open(img_path)
		#中の計算部分は貼る位置のピクセル調整。あまり気にしなくて良い
		img_transformed = self.transform(img, self.phase)  # torch.Size([3, 224, 224])
		label = img_path.split("/")
		label = self.classes[label[6]]
		
		return img_transformed, label

if __name__ == "__main__":
	config = configurations["config"]
	class_name = configurations["class_name"]
	batch_size = config["BATCH_SIZE"]
	size = config["IMAGE_SIZE"]
	mean = config["MEAN"]
	std = config["STD"]
	classes = config["CLASSES"]
	cross_classes = config["CROSS_CLASSES"]
	class_list = make_datapath_list(cross_classes)
	class_dataset = []
	class_dataloader = []
	
	for row in class_list:
		class_dataset.append(Dataset(
			file_list=row, transform=ImageTransform(size, mean, std), phase='train'))
	index = 0
	print(class_dataset[0].__getitem__(index)[0].size())
	print(class_dataset[0].__getitem__(index)[1])

# DataLoaderを作成
	
	for row in class_dataset:
		class_dataloader.append(torch.utils.data.DataLoader(
			row, batch_size=batch_size, shuffle=True))
	# 動作確認
		batch_iterator = iter(class_dataloader[0])  # イテレータに変換
		inputs, labels = next(batch_iterator)
		inputs, labels = next(batch_iterator)
		print(inputs.size()) #batch_size,color,height,weight
		print(labels)
		for i,row in enumerate(class_list):
			pd.DataFrame(row).to_csv("data/"+str(i)+".csv",index=False,header=False)
		f = pd.read_csv("data/0.csv")
	