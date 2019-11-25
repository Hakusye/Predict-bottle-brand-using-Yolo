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

#import tensorflow as tf
class BaseTransform():
    def __init__(self,resize,mean,std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
    def __call__(self,img):
        return self.base_transform(img)


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_

### ここに書き込処理が書いてある!
def write(x,img,transform):
#def write(x, img,net,transform):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    #print(cls)
    if(cls != 39):
        return img
    label = "{0}".format(classes[cls])
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    img1 =  img[int(x[2]):int(x[4]),int(x[1]):int(x[3])]
    #label = cnn_predict_bottle(img1,net,transform)
    label  = haming_predict_bottle(img1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    #cv2.putText(img, label, (c1[0], c1[1] + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

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
    label = ["koicha","ayataka_brown","calpis","namacha","natural_green",
            "tropicana","ooi_ochaa","coca_cola","ayataka","dekavita",
            "pokari","iemon","genmai","koicha"]
    
    net = net.to("cuda")
    net.eval()
    #img = torch.from_numpy(img)
    img = Image.fromarray(np.uint8(img))
    img = transform(img).unsqueeze_(0)
    img = img.to("cuda")
    results = net(img).to("cuda")
    _, predicted = torch.max(results.data, 1)
    return label[predicted]

def arg_parse():
    #Parse arguements to the detect module
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()


if __name__ == '__main__':
    #alr_train_path = "../bottle_predict/weights/ResNet50_batch32_epoch4.pth"
    #net = models.resnet50(pretrained=False)
    #net.fc = nn.Linear(2048,16)##ボトル2種類
    #net = net.to("cuda")
    #net.load_state_dict(torch.load(alr_train_path))
    cnt = 0
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    transform = BaseTransform(224,(0.485,0.456,0.406),(0.229,0.224,0.225))
    CUDA = torch.cuda.is_available()
    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    cap = cv2.VideoCapture(0)
    if(cap.isOpened() and cv2.cuda.getCudaEnabledDeviceCount()):
        print("Gpu")
    else:
        print("No Gpu")

    if CUDA:
        model.cuda()
    
    model(get_test_input(inp_dim, CUDA), CUDA)

    model.eval()
    
   # videofile = args.video
    
  #  cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
         
        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            
            with torch.no_grad():   
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                orig_im = cv2.resize(orig_im,(int(orig_im.shape[1]/4),int(orig_im.shape[0]/4)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
            
            #orig_im = cv2.resize(orig_im,(int(orig_im.shape[1]/2),int(orig_im.shape[0]/2)))
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
            
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
            
            output[:,1:5] /= scaling_factor
    
            for i in range(output.shape[0]):
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            ### 枠で囲うところ。最終的には付けたい
            #list(map(lambda x: write(x, orig_im,net,transform), output))
            list(map(lambda x: write(x, orig_im,transform), output))
            #print(im3)
            cv2.imshow("frame", orig_im)
            cnt+=1
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            
        else:
            break
    
###output[個体番号][左上x][左上y][右下x][右下y][class]
###bottleはclass39
###cv2.rectange(img,左上,右上,color,num)
    
    

