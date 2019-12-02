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
import cv2

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
	alr_train_path = "weights/ResNet50_classes14_epoch6.pth"
	transform = ImageTransform(size,mean,std)
	
	model = models.resnet50(pretrained=False)
	model.fc = nn.Linear(2048,classes)
	model.load_state_dict(torch.load(alr_train_path))
	#val_list,train_list = make_datapath_list()
	img = cv2.imread("../self_images_val/iemon/39_32.png")
	#cv2.imshow("test",img)
	cv2.waitKey(0)
	img = Image.fromarray(np.uint8(img))
	img = transform(img).unsqueeze_(0)
	#img = img.to("cuda")
	results = model(img)#.to("cuda")
	_,predicted = torch.max(results.data,1)
	print("ans:"+configurations["rev_class_name"][predicted])
	


