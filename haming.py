
from PIL import Image
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import cv2
from bottle_predict.config import configurations

def haming(target_path):
    searh_path = "/home/deepstation/Shirae/MYolo/haming_images/"
    size = 224/4
    target_dist = average_hash(target_path,size)
    images = glob.glob(os.path.join(searh_path,"*.png"))
    min_result = 10000000
    for i, fname in enumerate(images):
        dist = average_hash(fname, size)
        diff = hamming_dist(target_dist, dist) / 256
        if diff < min_result:
            min_result = diff
            result = fname
    return result,min_result


def average_hash(target_file, size):
    img = Image.open(target_file)
    img = img.resize((int(size), int(size)), Image.ANTIALIAS)  # 変換モードをRGBへ。
    px = np.array(img.getdata()).reshape((int(size), int(size), 3))  #リサイズの形状を3次元に変換
    avg = px.mean()
    px = 1 * (px > avg)
    return px

def hamming_dist(a, b):    
    a = a.reshape(1, -1)  # 1次元に変換
    b = b.reshape(1, -1)  # 1次元に変換
    dist = (a != b).sum()  # 要素が異なる部分の合計値を計算
    return dist

 

#確認
if __name__ == "__main__":
    target_file = "haming_images/ayataka.png"
    img = Image.open(target_file)
    plt.figure(figsize=(128,128))
    ans_img_path,result = haming(target_file)
    img = Image.open(ans_img_path,'r')
    #plt.imshow(np.asarray(img))
    print(result)
    img.show()

