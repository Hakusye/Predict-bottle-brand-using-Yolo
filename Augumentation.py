#!/usr/bin/env python3
import Augmentor
from bottle_predict.config import configurations
import os.path as osp
from pprint import pprint
import glob
images_dir = configurations["config"]["ROOT_IMAGES_PATH"]
target_path = osp.join(images_dir + "*/")
#target_path = osp.join(images_dir + '*/*' + configurations["config"]["EXT"])
#for i in glob.glob(target_path):
#	print(i)
for images_dir in glob.glob(target_path):
    p = Augmentor.Pipeline(images_dir)
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=5)
    p.random_contrast(probability=0.5, min_factor=0.4, max_factor=0.8)
    p.random_color(probability=0.5, min_factor=0.4, max_factor=0.8)
    p.rotate90(probability=0.5)
    p.rotate270(probability=0.5)
    p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.8)
    p.flip_top_bottom(probability=0.3)
    p.sample(200)
