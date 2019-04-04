'''
ReNet-Keras
Code written by: Xiaoqing Liu
If you use significant portions of this code or the ideas from our paper, please cite it :)

'''
from __future__ import print_function
# import threading
import logging
import os
from os.path import join, basename
from os import makedirs

import cv2

import numpy as np
from numpy.random import rand, shuffle
from PIL import Image

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

plt.ioff()

debug = 0    

def read_labeled_image_list(data_dir, data_list):
    f = open(data_dir+'/'+data_list, 'r')

    path_total=[]
    path = []
    for line in f:
        try:
            image, mask = line[:-1].split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = line.strip("\n")

        path_1 = os.path.join(data_dir+'/Images', image+'.png')
        path.append(path_1)
        path_2 = os.path.join(data_dir+'/Labels', mask+'.png')
        path.append(path_2) 
        path_total.append(path)
    np.random.shuffle(path_total)
    return path_total



def train_total(data_dir, data_list='train.txt'):
    
    path_total=read_labeled_image_list(data_dir, data_list)
    images = []
    masks = []
    #excerpt = [next(path_total) for i in range(batch_size)]
    # for start_idx in range(0, len(path_total) - batch_size + 1, batch_size):
        # excerpt = path_total[start_idx:start_idx + batch_size]
    for i in range(len(path_total)):
        img_path = path_total[i][0]
        img = cv2.imread(img_path)
        image = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR)
        images.append(image)

        lab_path = path_total[i][1]
        mas = cv2.imread(lab_path)
        mask = cv2.resize(mas, (32, 32), interpolation=cv2.INTER_LINEAR)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_gray_new = mask_gray[:, :, np.newaxis]
        masks.append(mask_gray_new)

        images_total=np.array(images)
        masks_total=np.array(masks)

        images_batch_0=images_total[:,:, :, 0]
        images_batch_new = images_batch_0[:,:, :, np.newaxis]
    #return images_total,masks_total
    return images_total,masks_total




def generate_val_batches(data_dir, batch_size, data_list='val.txt', shuffle=True):
    path_total=read_labeled_image_list(data_dir+'/test', data_list)
    images = []
    masks = []
    for start_idx in range(0, len(path_total) - batch_size + 1, batch_size):
        excerpt = path_total[start_idx:start_idx + batch_size]
        for i in range(len(excerpt)):
            img_path = excerpt[i][0]
            img = cv2.imread(img_path)
            image = cv2.resize(img,(512,512),interpolation=cv2.INTER_LINEAR)
            images.append(image)   
         
            lab_path = excerpt[i][1]
            mas = cv2.imread(lab_path)
            mask = cv2.resize(mas,(512,512),interpolation=cv2.INTER_LINEAR)
            mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            mask_gray_new=mask_gray[:,:,np.newaxis]
            masks.append(mask_gray_new)

            images_batch=np.array(images)
            masks_batch=np.array(masks)

            images_batch_0=images_batch[:,:, :, 0]
            images_batch_new = images_batch_0[:,:, :, np.newaxis]
        yield ([images_batch, masks_batch],[masks_batch,masks_batch*images_batch_new])
        

