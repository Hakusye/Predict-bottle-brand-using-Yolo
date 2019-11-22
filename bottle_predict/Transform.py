import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
#%matplotlib inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import models, transforms

class BaseTransform():
	def __init__(self, resize, mean, std):
		self.base_transform = transforms.Compose([
				transforms.Resize(resize),
				transforms.CenterCrop(resize),
				transforms.ToTensor(), 
				transforms.Normalize(mean, std)
		])

	def __call__(self, img):
		return self.base_transform(img)

class ImageTransform():
	def __init__(self, resize, mean, std):
		self.data_transform = {
		'train': transforms.Compose([
		transforms.RandomResizedCrop(
					resize, scale=(0.5, 1.0)),  
					transforms.RandomHorizontalFlip(), 
					transforms.ToTensor(),  
					transforms.Normalize(mean, std) 
			]),
			'val': transforms.Compose([
					transforms.Resize(resize), 
					transforms.CenterCrop(resize), 
					transforms.ToTensor(), 
					transforms.Normalize(mean, std) 
			])
		}

	def __call__(self, img, phase='train'):
		return self.data_transform[phase](img)



