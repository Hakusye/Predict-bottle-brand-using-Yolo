
from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse
from PIL import Image
import haming
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from config import configurations
from bottle_predict.Transform import *
from attach_label import *


### haming距離による画像分類
def haming_predict_bottle(img):
    img = img[:,:,::-1].copy()
    img = Image.fromarray(img)
    ans_path,result = haming.haming(img)
    label = ans_path.split("/")
    label = label[6][:-4]
    return label



### cnnの飲み物の画像分類
def cnn_predict_bottle(img,net,transform):
    label = configurations["rev_class_name"]
    size = configurations["config"]["IMAGE_SIZE"]
    net = net.to("cuda")
    net.eval()
    #img = torch.from_numpy(img)
    img = Image.fromarray(np.uint8(img))
    img = transform(img).unsqueeze_(0)
    img = img.to("cuda")
    results = net(img).to("cuda")
    per, predicted = torch.max(results.data, 1)
    return label[predicted]
