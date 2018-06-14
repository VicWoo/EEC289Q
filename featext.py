#This module is to resize images and extract features from them using pre-trained dnns models

import mxnet as mx
from mxnet import init, gluon, nd, autograd, image
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import h5py
import os
from glob import glob
import matplotlib.pyplot as plt

ctx = mx.cpu()
data_dir = "data" 
imageSize_resnet =  331 #This is the input size required by both NASNet and InceptRes v2
n = len(glob(os.path.join('.', data_dir, "Images", "*", "*.jpg"))) #Number of training images

#-------------------------------------center cropping function-------------------------------
def centeredCrop(img, new_height, new_width):
   width =  np.size(img,1)
   height =  np.size(img,0)
   left = int(np.round((width - new_width)/2.))
   top = int(np.round((height - new_height)/2.))
   right = int(np.round((width + new_width)/2.))
   bottom = int(np.round((height + new_height)/2.))
   cImg = img[top:bottom, left:right]
   return cImg

#------------------------------------ Model and image augmentation parameters----------------
net = models.get_model('inceptionv3', pretrained=True, ctx=ctx) #enter name of model as given on gluon model zoo
mean = np.array([0.485, 0.456, 0.406]) #Both mean and std are given by mxnet settings
std = np.array([0.229, 0.224, 0.225])  #Normalization of images

#--------------------------------------batch (=128) feature extraction--------------------------------------
features = []
for j in tqdm(range(0,80)):
    i = 0
    temp = nd.zeros((128, 3, imageSize_resnet, imageSize_resnet))
    for file_name in glob(os.path.join(data_dir, "for_train", "*", "*.jpg"))[128*j:128*(j+1)]:
        img = cv2.imread(file_name)
        # Resizing and center cropping to reduce object deformation resulting from using resize only
        if img.shape[0]>img.shape[1]:
           img_224 = ((cv2.resize(img, (imageSize_resnet,round(imageSize_resnet*img.shape[0]/img.shape[1])))[:,:,::-1] \
                    / 255.0 - mean) / std)
           if img_224.shape[1]<imageSize_resnet:
              img_224 = cv2.resize(img,(imageSize_resnet,imageSize_resnet))[:,:,::-1] 
           else:
              img_224 = centeredCrop(img_224,imageSize_resnet,imageSize_resnet)
           print(img_224.shape,1,'b')
        else:
           img_224 = ((cv2.resize(img, (round(imageSize_resnet*img.shape[1]/img.shape[0]),imageSize_resnet))[:,:,::-1] \
                    / 255.0 - mean) / std)
           print(img_224.shape,2,'a')
           if img_224.shape[0]<imageSize_resnet:
              img_224 = cv2.resize(img,(imageSize_resnet,imageSize_resnet))[:,:,::-1] 
           else:
              img_224 = centeredCrop(img_224,imageSize_resnet,imageSize_resnet)
           print(img_224.shape,2,'b')
        plt.imshow(img_224)
        plt.show()
        img_224_compare = cv2.resize(img,(imageSize_resnet,imageSize_resnet))[:,:,::-1]
        plt.imshow(img_224_compare)
        plt.show()
        img_224 = img_224.transpose((2, 0, 1))
        temp[i] = nd.array((img_224))
        nd.waitall()
        i += 1
    #Last batch
    if j == 79:
        temp = temp[0:110]
    data_iter_224 = gluon.data.DataLoader(gluon.data.ArrayDataset(temp), batch_size=128)
    for data in data_iter_224:
        feature = net.features(data.as_in_context(mx.cpu()))
        feature = gluon.nn.Flatten()(feature)
        features.append(feature.as_in_context(mx.cpu()))
    nd.waitall()
features = nd.concat(*features, dim=0)

#----------------------------------------------save features-----------------------------------------------
print(features)
print(features.shape)
nd.save(os.path.join(data_dir, 'features_incep_384.nd'), features) #change filename 'features_<modelname>_<imagedimension>.nd'


