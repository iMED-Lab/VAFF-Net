#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :datasets.py
@Description :Dataset for multi task, data could be choosen from vessel,junctions and FAZ
@Time        :2020/10/30 14:45:26
@Author      :Jinkui Hao
@Version     :1.0
'''

import torch.utils.data as data
import torch
import numpy as np
import os
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
from scipy import misc
import scipy.io as sio
import csv
import nibabel as nib
from torch.utils.data import DataLoader
import math
from utils.tools import *


def random_crop(data, junction1, junction2, FAZ, vessel, crop_size):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(data, output_size=crop_size)
    data = TF.crop(data, i, j, h, w)
    junction1 = TF.crop(junction1, i, j, h, w)
    junction2 = TF.crop(junction2, i, j, h, w)
    FAZ = TF.crop(FAZ, i, j, h, w)
    vessel = TF.crop(vessel, i, j, h, w)

    return data, junction1, junction2, FAZ, vessel

def random_crop_8(data, junction1_map, junction2_map, junction1_det, junction2_det, FAZ, vessel, crop_size):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(data, output_size=crop_size)
    data = TF.crop(data, i, j, h, w)
    junction1_map = TF.crop(junction1_map, i, j, h, w)
    junction2_map = TF.crop(junction2_map, i, j, h, w)
    junction1_det = TF.crop(junction1_det, i, j, h, w)
    junction2_det = TF.crop(junction2_det, i, j, h, w)
    FAZ = TF.crop(FAZ, i, j, h, w)
    vessel = TF.crop(vessel, i, j, h, w)

    return data, junction1_map, junction2_map, junction1_det, junction2_det, FAZ, vessel

def random_crop_10(img1,img2,img3, junction1_map, junction2_map, junction1_det, junction2_det, FAZ, vessels, crop_size):
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(img1, output_size=crop_size)
    img1 = TF.crop(img1, i, j, h, w)
    img2 = TF.crop(img2, i, j, h, w)
    img3 = TF.crop(img3, i, j, h, w)
    junction1_map = TF.crop(junction1_map, i, j, h, w)
    junction2_map = TF.crop(junction2_map, i, j, h, w)
    junction1_det = TF.crop(junction1_det, i, j, h, w)
    junction2_det = TF.crop(junction2_det, i, j, h, w)
    FAZ = TF.crop(FAZ, i, j, h, w)
    #vessel = TF.crop(vessel, i, j, h, w)

    vessels[0] = TF.crop(vessels[0], i, j, h, w)
    vessels[1] = TF.crop(vessels[1], i, j, h, w)
    vessels[2] = TF.crop(vessels[2], i, j, h, w)

    return img1,img2,img3, junction1_map, junction2_map, junction1_det, junction2_det, FAZ, vessels

def img_transforms(img, junction1, junction2, FAZ, vessel, crop_size):
    trans_pad = transforms.Pad(15)
    trans_tensor = transforms.ToTensor()

    img, junction1, junction2, FAZ, vessel = trans_pad(img), trans_pad(junction1),\
                                trans_pad(junction2),trans_pad(FAZ),trans_pad(vessel)

    img, junction1, junction2, FAZ, vessel = random_crop(img, junction1, junction2, FAZ, vessel, crop_size)

    img, junction1, junction2, FAZ, vessel = trans_tensor(img), trans_tensor(junction1),\
                                trans_tensor(junction2),trans_tensor(FAZ),trans_tensor(vessel)

    return img, junction1, junction2, FAZ, vessel

def img_transforms_8(img, junction1_map, junction2_map, junction1_det, junction2_det, FAZ, vessel, crop_size):
    trans_pad = transforms.Pad(15)
    trans_tensor = transforms.ToTensor()

    img, junction1_map, junction2_map, junction1_det, junction2_det, FAZ, vessel = trans_pad(img), trans_pad(junction1_map),trans_pad(junction2_map),\
                                trans_pad(junction1_det),trans_pad(junction2_det),trans_pad(FAZ),trans_pad(vessel)

    img, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessel = random_crop_8(img, junction1_map, junction2_map, junction1_det, junction2_det, FAZ, vessel, crop_size)

    img, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessel = trans_tensor(img), trans_tensor(junction1_map),trans_tensor(junction2_map),\
                                trans_tensor(junction1_det),trans_tensor(junction2_det),trans_tensor(FAZ),trans_tensor(vessel)

    return img, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessel

def img_transforms_10(img1,img2,img3, junction1_map, junction2_map, junction1_det, junction2_det, FAZ, vessels, crop_size):
    if random.random() > 0.5:
        img1,img2,img3, junction1_map, junction2_map, junction1_det, junction2_det, FAZ = TF.hflip(img1),TF.hflip(img2),TF.hflip(img3), \
                                TF.hflip(junction1_map),TF.hflip(junction2_map),\
                                TF.hflip(junction1_det),TF.hflip(junction2_det),TF.hflip(FAZ)
    
        vessels[0] = TF.hflip(vessels[0])
        vessels[1] = TF.hflip(vessels[1])
        vessels[2] = TF.hflip(vessels[2])
        
    if random.random() > 0.5:
        img1,img2,img3, junction1_map, junction2_map, junction1_det, junction2_det, FAZ = TF.vflip(img1),TF.vflip(img2),TF.vflip(img3), \
                                TF.vflip(junction1_map),TF.vflip(junction2_map),\
                                TF.vflip(junction1_det),TF.vflip(junction2_det),TF.vflip(FAZ)
    
        vessels[0] = TF.vflip(vessels[0])
        vessels[1] = TF.vflip(vessels[1])
        vessels[2] = TF.vflip(vessels[2])

    trans_pad = transforms.Pad(10)
    trans_tensor = transforms.ToTensor()

    img1,img2,img3, junction1_map, junction2_map, junction1_det, junction2_det, FAZ = trans_pad(img1),trans_pad(img2),trans_pad(img3), \
                                trans_pad(junction1_map),trans_pad(junction2_map),\
                                trans_pad(junction1_det),trans_pad(junction2_det),trans_pad(FAZ)
    
    vessels[0] = trans_pad(vessels[0])
    vessels[1] = trans_pad(vessels[1])
    vessels[2] = trans_pad(vessels[2])

    img1,img2,img3, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessels = random_crop_10(img1,img2,img3, junction1_map, junction2_map, junction1_det, junction2_det, FAZ, vessels, crop_size)

    img1,img2,img3, junction1_map,junction2_map, junction1_det,junction2_det, FAZ = trans_tensor(img1),trans_tensor(img2),trans_tensor(img3), \
                                trans_tensor(junction1_map),trans_tensor(junction2_map),\
                                trans_tensor(junction1_det),trans_tensor(junction2_det),trans_tensor(FAZ)

    vessels[0] = trans_tensor(vessels[0])
    vessels[1] = trans_tensor(vessels[1])
    vessels[2] = trans_tensor(vessels[2])

    return img1,img2,img3, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessels

class datasetMT(data.Dataset):
    def __init__(self, root, task=['vessel','junctions','FAZ'],sigma=1.5, isTraining = True):
        #@task: could be vessel, junction, FAZ or All
        self.root = root
        self.taskName = task
        self.isTraining = isTraining
        self.name = ''
        self.sigma = sigma
        self.imgPath = self.getImgPath(root,isTraining)

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        image = Image.open(imgPath)
        image = image.convert('RGB')
        junction1 = []
        junction2 = []
        FAZ = []
        vessel = []
        #label
        gtPath = os.path.join(self.root, imgPath.split('/')[-4])
        self.name = imgPath.split('/')[-1]
        
        for task in self.taskName:
            
            if task == 'junctions':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'mat')
                juncMat = sio.loadmat(taskPath)
                junction1,junction2,junctions = self.getJuncMap(juncMat,image.height)
            elif task == 'FAZ':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'png')
                FAZ = Image.open(taskPath)
                FAZ = FAZ.convert('L')
            else:
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'png')
                vessel = Image.open(taskPath)
                vessel = vessel.convert('L')

        
        # augumentadatasetMT_3Intion,transform
        if self.isTraining:
            rotate = 10
            angel = random.randint(-rotate, rotate)
            image = image.rotate(angel)
            junction1 = junction1.rotate(angel)
            junction2 = junction2.rotate(angel)
            FAZ = FAZ.rotate(angel)
            vessel = vessel.rotate(angel)

            image, junction1, junction2, FAZ, vessel = \
                img_transforms(image, junction1, junction2, FAZ, vessel, (304,304))

        else: 
            trans_tensor = transforms.ToTensor()

            image, junction1, junction2, FAZ, vessel = trans_tensor(image), trans_tensor(junction1),\
                                trans_tensor(junction2),trans_tensor(FAZ),trans_tensor(vessel)

        background = torch.ones_like(junction1) - junction1 - junction2
        #imgBack = background.numpy()
        #cv2.imwrite('back.jpg', imgBack[0,:,:]*255)
        junction = torch.cat([junction1,junction2,background], 0)

        return image, junction, FAZ, vessel
        
    def __len__(self):
        return len(self.imgPath)

    def getImgPath(self,root, isTraining):
        
        if isTraining:
            imgPath = os.path.join(root,'train', 'img', 'WRCC')
        else:
            imgPath = os.path.join(root,'test', 'img', 'WRCC')

        items = []
        imgList = os.listdir(imgPath)
        for name in imgList:
            pathRelative = os.path.join(imgPath,name)
            items.append(pathRelative)

        return items

    def getJuncMap(self,mat,shape):
        junc1 = np.zeros((shape,shape))
        junc2 = np.zeros((shape,shape))
        juncs = np.zeros((shape,shape))

        for point in mat['bifurcation']:
            junc1 = put_heatmap(junc1, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)

        for point in mat['crossing']:
            junc2 = put_heatmap(junc2, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)
       
        #cv2.imwrite('junction_2.0.png',junc1*255)

        return Image.fromarray(junc1), Image.fromarray(junc2), Image.fromarray(juncs)

    def getFileName(self):
        return self.name

class datasetMT_detect(data.Dataset):
    #基于检测做junction定位，返回区域坐标和类别信息
    def __init__(self, root, task=['vessel','junctions','FAZ'],size=8, isTraining = True):
        #@task: could be vessel, junction, FAZ or All
        self.root = root
        self.taskName = task
        self.isTraining = isTraining
        self.name = ''
        self.girdSize = size
        self.imgPath = self.getImgPath(root,isTraining)

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        image = Image.open(imgPath)
        image = image.convert('RGB')
        junction1 = []
        junction2 = []
        FAZ = []
        vessel = []
        #label
        gtPath = os.path.join(self.root, imgPath.split('/')[-4])
        self.name = imgPath.split('/')[-1]
        
        for task in self.taskName:
            
            if task == 'junctions':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'mat')
                juncMat = sio.loadmat(taskPath)
                junction1,junction2 = drawPoint(juncMat,size=(304,304))
                
            elif task == 'FAZ':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'png')
                FAZ = Image.open(taskPath)
                FAZ = FAZ.convert('L')
            else:
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'png')
                vessel = Image.open(taskPath)
                vessel = vessel.convert('L')

        # augumentation,transform
        if self.isTraining:
            rotate = 15
            angel = random.randint(-rotate, rotate)
            image = image.rotate(angel)
            junction1 = junction1.rotate(angel)
            junction2 = junction2.rotate(angel)
            FAZ = FAZ.rotate(angel)
            vessel = vessel.rotate(angel)

            image, junction1, junction2, FAZ, vessel = \
                img_transforms(image, junction1, junction2, FAZ, vessel, (304,304))
        else: 
            trans_tensor = transforms.ToTensor()
            image, junction1, junction2, FAZ, vessel = trans_tensor(image), trans_tensor(junction1),\
                                trans_tensor(junction2),trans_tensor(FAZ),trans_tensor(vessel)


        _,_, labelAll6, pointImg = genDetectLabel(junction1.squeeze().numpy(), \
                                                            junction2.squeeze().numpy(), self.girdSize)
        junction = labelAll6

        # imgS = image.squeeze().numpy()
        # cv2.imwrite('image.jpg', imgS[0,:,:]*255+pointImg)
        # cv2.imwrite('vvv.jpg', pointImg)

        return image, junction, FAZ, vessel
        
    def __len__(self):
        return len(self.imgPath)

    def getImgPath(self,root, isTraining):
        
        if isTraining:
            imgPath = os.path.join(root,'train', 'img','WRCC')
        else:
            imgPath = os.path.join(root,'test', 'img','WRCC')

        items = []
        imgList = os.listdir(imgPath)
        for name in imgList:
            pathRelative = os.path.join(imgPath,name)
            items.append(pathRelative)

        return items

    def getJuncMap(self,mat,shape):
        junc1 = np.zeros((shape,shape))
        junc2 = np.zeros((shape,shape))
        juncs = np.zeros((shape,shape))

        for point in mat['bifurcation']:
            junc1 = put_heatmap(junc1, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)

        for point in mat['crossing']:
            junc2 = put_heatmap(junc2, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)
       
        #cv2.imwrite('junction_2.0.png',junc1*255)

        return Image.fromarray(junc1), Image.fromarray(junc2), Image.fromarray(juncs)

    def getFileName(self):

        return self.name

class datasetMT_3In(data.Dataset):
    #检测和heatmap的代码
    def __init__(self, root, task=['vessel','junctions','FAZ'],sigma=1.5,size = 4, isTraining = True, imgSize = 304):
        #@task: could be vessel, junction, FAZ or All
        self.root = root
        self.taskName = task
        self.isTraining = isTraining
        self.name = ''
        self.sigma = sigma
        self.girdSize = size
        self.imgPath = self.getImgPath(root,isTraining)
        self.imgSize = imgSize

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        gtPath = os.path.join(self.root, imgPath.split('/')[-3])
        self.name = imgPath.split('/')[-1]

        if '.png' in self.name:
            image_w = Image.open(os.path.join(imgPath[:-len(self.name)-1],'WRCC',self.name))
            image_w = image_w.convert('RGB')
            image_d = Image.open(os.path.join(imgPath[:-len(self.name)-1],'DCC',self.name[:-3]+'tif'))
            image_d = image_d.convert('RGB')
            image_s = Image.open(os.path.join(imgPath[:-len(self.name)-1],'SCC',self.name[:-3]+'tif'))
            image_s = image_s.convert('RGB')
        else:
            image_w = Image.open(os.path.join(imgPath[:-len(self.name)-1],'WRCC',self.name))
            image_w = image_w.convert('RGB')
            image_d = Image.open(os.path.join(imgPath[:-len(self.name)-1],'DCC',self.name))
            image_d = image_d.convert('RGB')
            image_s = Image.open(os.path.join(imgPath[:-len(self.name)-1],'SCC',self.name))
            image_s = image_s.convert('RGB')

        

        junction1 = []
        junction2 = []
        FAZ = []
        vessel = []
      
        for task in self.taskName:
            
            if task == 'junctions':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'mat')
                juncMat = sio.loadmat(taskPath)
                junction1_map,junction2_map,_ = self.getJuncMap(juncMat,image_w.height)

                junction1_det,junction2_det = drawPoint(juncMat,size=(self.imgSize,self.imgSize))
            elif task == 'FAZ':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'png')
                FAZ = Image.open(taskPath)
                FAZ = FAZ.convert('L')
            else:
                vessels = []
                vessel_w = Image.open(os.path.join(gtPath, 'gt', task, self.name[:-3]+'png'))
                vessel_w = vessel_w.convert('L')
                vessels.append(vessel_w)

                vessels.append(vessel_w)

                vessels.append(vessel_w)

        # augumentation,transform
        if self.isTraining:

            gamma_v = round(np.random.uniform(0.7,1.9),2)
            image_w = TF.adjust_gamma(img=image_w, gamma = gamma_v)
            image_d = TF.adjust_gamma(img=image_d, gamma = gamma_v)
            image_s = TF.adjust_gamma(img=image_s, gamma = gamma_v)
            rotate = 10

            angel = random.randint(-rotate, rotate)
            image_w = image_w.rotate(angel)
            image_d = image_d.rotate(angel)
            image_s = image_s.rotate(angel)

            junction1_map = junction1_map.rotate(angel)
            junction2_map = junction2_map.rotate(angel)

            junction1_det = junction1_det.rotate(angel)
            junction2_det = junction2_det.rotate(angel)

            FAZ = FAZ.rotate(angel)
            vessels[0] = vessels[0].rotate(angel)
            vessels[1] = vessels[1].rotate(angel)
            vessels[2] = vessels[2].rotate(angel)

            image_w,image_d,image_s, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessels = \
                img_transforms_10(image_w,image_d,image_s, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessels, (self.imgSize,self.imgSize))

        else: 
            trans_tensor = transforms.ToTensor()

            image_w,image_d,image_s, junction1_map,junction2_map, junction1_det,junction2_det, FAZ = trans_tensor(image_w),trans_tensor(image_d),trans_tensor(image_s), \
                                trans_tensor(junction1_map),trans_tensor(junction2_map),\
                                trans_tensor(junction1_det),trans_tensor(junction2_det),trans_tensor(FAZ)
            vessels[0] = trans_tensor(vessels[0])
            vessels[1] = trans_tensor(vessels[1])
            vessels[2] = trans_tensor(vessels[2])


        background_map = torch.ones_like(junction1_map) - junction1_map - junction2_map
        #imgBack = background.numpy()
        #cv2.imwrite('back.jpg', imgBack[0,:,:]*255)
        junction_map = torch.cat([junction1_map,junction2_map,background_map], 0)

        _,_, labelAll6, pointImg = genDetectLabel(junction1_det.squeeze().numpy(), \
                                                            junction2_det.squeeze().numpy(), self.girdSize)
        junction_det = labelAll6


        return (image_w,image_d,image_s), junction_map,junction_det, FAZ, (vessels[0],vessels[1],vessels[2])
        
    def __len__(self):
        return len(self.imgPath)

    def getImgPath(self,root, isTraining):
        
        if isTraining:
            imgPath = os.path.join(root,'train', 'img','WRCC')
        else:
            imgPath = os.path.join(root,'test', 'img','WRCC')

        items = []
        imgList = os.listdir(imgPath)
        for name in imgList:
            pathRelative = os.path.join(imgPath[:-4],name)
            items.append(pathRelative)

        return items

    def getJuncMap(self,mat,shape):
        junc1 = np.zeros((shape,shape))
        junc2 = np.zeros((shape,shape))
        juncs = np.zeros((shape,shape))

        for point in mat['bifurcation']:
            junc1 = put_heatmap(junc1, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)

        for point in mat['crossing']:
            junc2 = put_heatmap(junc2, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)
       
        #cv2.imwrite('junction_2.0.png',junc1*255)

        return Image.fromarray(junc1), Image.fromarray(junc2), Image.fromarray(juncs)

    def getFileName(self):
        return self.name

class datasetSig_3In(data.Dataset):
    #单任务对比实验，输入为3个通道合并
    def __init__(self, root, task=['vessel','junctions','FAZ'],sigma=1.5,size = 4, isTraining = True, imgSize = 304):
        #@task: could be vessel, junction, FAZ or All
        self.root = root
        self.taskName = task
        self.isTraining = isTraining
        self.name = ''
        self.sigma = sigma
        self.girdSize = size
        self.imgPath = self.getImgPath(root,isTraining)
        self.imgSize = imgSize

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        gtPath = os.path.join(self.root, imgPath.split('/')[-3])
        self.name = imgPath.split('/')[-1]
        

        if '.png' in self.name:
            image_w = Image.open(os.path.join(imgPath[:-len(self.name)-1],'WRCC',self.name))
            image_w = image_w.convert('L')
            image_d = Image.open(os.path.join(imgPath[:-len(self.name)-1],'DCC',self.name[:-3]+'tif'))
            image_d = image_d.convert('L')
            image_s = Image.open(os.path.join(imgPath[:-len(self.name)-1],'SCC',self.name[:-3]+'tif'))
            image_s = image_s.convert('L')
        else:
            image_w = Image.open(os.path.join(imgPath[:-len(self.name)-1],'WRCC',self.name))
            image_w = image_w.convert('L')
            image_d = Image.open(os.path.join(imgPath[:-len(self.name)-1],'DCC',self.name))
            image_d = image_d.convert('L')
            image_s = Image.open(os.path.join(imgPath[:-len(self.name)-1],'SCC',self.name))
            image_s = image_s.convert('L')

        

        junction1 = []
        junction2 = []
        FAZ = []
        vessel = []
      
        for task in self.taskName:
            
            if task == 'junctions':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'mat')
                juncMat = sio.loadmat(taskPath)
                junction1_map,junction2_map,_ = self.getJuncMap(juncMat,image_w.height)

                junction1_det,junction2_det = drawPoint(juncMat,size=(self.imgSize,self.imgSize))
            elif task == 'FAZ':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-3]+'png')
                FAZ = Image.open(taskPath)
                FAZ = FAZ.convert('L')
            else:
                vessels = []
                vessel_w = Image.open(os.path.join(gtPath, 'gt', task, self.name[:-3]+'png'))
                vessel_w = vessel_w.convert('L')
                vessels.append(vessel_w)

                vessels.append(vessel_w)

                vessels.append(vessel_w)

        # augumentation,transform
        if self.isTraining:

            gamma_v = round(np.random.uniform(0.7,1.9),2)
            image_w = TF.adjust_gamma(img=image_w, gamma = gamma_v)
            image_d = TF.adjust_gamma(img=image_d, gamma = gamma_v)
            image_s = TF.adjust_gamma(img=image_s, gamma = gamma_v)

            rotate = 10

            angel = random.randint(-rotate, rotate)
            image_w = image_w.rotate(angel)
            image_d = image_d.rotate(angel)
            image_s = image_s.rotate(angel)

            junction1_map = junction1_map.rotate(angel)
            junction2_map = junction2_map.rotate(angel)

            junction1_det = junction1_det.rotate(angel)
            junction2_det = junction2_det.rotate(angel)

            FAZ = FAZ.rotate(angel)
            vessels[0] = vessels[0].rotate(angel)
            vessels[1] = vessels[1].rotate(angel)
            vessels[2] = vessels[2].rotate(angel)

            image_w,image_d,image_s, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessels = \
                img_transforms_10(image_w,image_d,image_s, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessels, (self.imgSize,self.imgSize))

        else: 
            trans_tensor = transforms.ToTensor()

            image_w,image_d,image_s, junction1_map,junction2_map, junction1_det,junction2_det, FAZ = trans_tensor(image_w),trans_tensor(image_d),trans_tensor(image_s), \
                                trans_tensor(junction1_map),trans_tensor(junction2_map),\
                                trans_tensor(junction1_det),trans_tensor(junction2_det),trans_tensor(FAZ)
            vessels[0] = trans_tensor(vessels[0])
            vessels[1] = trans_tensor(vessels[1])
            vessels[2] = trans_tensor(vessels[2])


        background_map = torch.ones_like(junction1_map) - junction1_map - junction2_map
        #imgBack = background.numpy()
        #cv2.imwrite('back.jpg', imgBack[0,:,:]*255)
        junction_map = torch.cat([junction1_map,junction2_map,background_map], 0)

        _,_, labelAll6, pointImg = genDetectLabel(junction1_det.squeeze().numpy(), \
                                                            junction2_det.squeeze().numpy(), self.girdSize)
        junction_det = labelAll6

        image = torch.cat([image_w,image_d,image_s], 0)

        return image, junction_map,junction_det, FAZ, vessels[0]
        
    def __len__(self):
        return len(self.imgPath)

    def getImgPath(self,root, isTraining):
        
        if isTraining:
            imgPath = os.path.join(root,'train', 'img','WRCC')
        else:
            imgPath = os.path.join(root,'test', 'img','WRCC')

        items = []
        imgList = os.listdir(imgPath)
        for name in imgList:
            pathRelative = os.path.join(imgPath[:-4],name)
            items.append(pathRelative)

        return items

    def getJuncMap(self,mat,shape):
        junc1 = np.zeros((shape,shape))
        junc2 = np.zeros((shape,shape))
        juncs = np.zeros((shape,shape))

        for point in mat['bifurcation']:
            junc1 = put_heatmap(junc1, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)

        for point in mat['crossing']:
            junc2 = put_heatmap(junc2, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)
       
        #cv2.imwrite('junction_2.0.png',junc1*255)

        return Image.fromarray(junc1), Image.fromarray(junc2), Image.fromarray(juncs)

    def getFileName(self):
        return self.name

class datasetMT_3In_zeiss(data.Dataset):
    #检测和heatmap的代码
    def __init__(self, root, task=['vessel','junctions','FAZ'],sigma=1.5,size = 4, isTraining = True):
        #@task: could be vessel, junction, FAZ or All
        self.root = root
        self.taskName = task
        self.isTraining = isTraining
        self.name = ''
        self.sigma = sigma
        self.girdSize = size
        self.imgPath = self.getImgPath(root,isTraining)

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        gtPath = os.path.join(self.root, imgPath.split('/')[-3])
        self.name = imgPath.split('/')[-1]
        
        image_w = Image.open(os.path.join(imgPath[:-len(self.name)-1],'WRCC',self.name))
        image_w = image_w.convert('RGB')
        image_d = Image.open(os.path.join(imgPath[:-len(self.name)-1],'DCC',self.name))
        image_d = image_d.convert('RGB')
        image_s = Image.open(os.path.join(imgPath[:-len(self.name)-1],'SCC',self.name))
        image_s = image_s.convert('RGB')

       
        trans_tensor = transforms.ToTensor()

        image_w,image_d,image_s = trans_tensor(image_w),trans_tensor(image_d),trans_tensor(image_s)

        return (image_w,image_d,image_s)
        
    def __len__(self):
        return len(self.imgPath)

    def getImgPath(self,root, isTraining):
        
        if isTraining:
            imgPath = os.path.join(root,'train', 'img','WRCC')
        else:
            imgPath = os.path.join(root,'test', 'img','WRCC')

        items = []
        imgList = os.listdir(imgPath)
        for name in imgList:
            pathRelative = os.path.join(imgPath[:-4],name)
            items.append(pathRelative)

        return items

    def getFileName(self):
        return self.name

class datasetMT_all(data.Dataset):
    #检测和heatmap的代码
    def __init__(self, root, task=['vessel','junctions','FAZ'],sigma=1.5,size = 4, isTraining = True):
        #@task: could be vessel, junction, FAZ or All
        self.root = root
        self.taskName = task
        self.isTraining = isTraining
        self.name = ''
        self.sigma = sigma
        self.girdSize = size
        self.imgPath = self.getImgPath(root,isTraining)

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        image = Image.open(imgPath)
        image = image.convert('RGB')
        junction1 = []
        junction2 = []
        FAZ = []
        vessel = []
        #label
        gtPath = os.path.join(self.root, imgPath.split('/')[-3])
        self.name = imgPath.split('/')[-1]
        
        for task in self.taskName:
            
            if task == 'junctions':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:2]+'.mat')
                juncMat = sio.loadmat(taskPath)
                junction1_map,junction2_map,_ = self.getJuncMap(juncMat,image.height)

                junction1_det,junction2_det = drawPoint(juncMat)
            elif task == 'FAZ':
                taskPath = os.path.join(gtPath, 'gt', task, self.name)
                FAZ = Image.open(taskPath)
                FAZ = FAZ.convert('L')
            else:
                taskPath = os.path.join(gtPath, 'gt', task, self.name)
                vessel = Image.open(taskPath)
                vessel = vessel.convert('L')

        # augumentation,transform
        if self.isTraining:
            rotate = 15
            angel = random.randint(-rotate, rotate)
            image = image.rotate(angel)

            junction1_map = junction1_map.rotate(angel)
            junction2_map = junction2_map.rotate(angel)

            junction1_det = junction1_det.rotate(angel)
            junction2_det = junction2_det.rotate(angel)

            FAZ = FAZ.rotate(angel)
            vessel = vessel.rotate(angel)

            image, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessel = \
                img_transforms_8(image, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessel, (304,304))

        else: 
            trans_tensor = transforms.ToTensor()

            image, junction1_map,junction2_map, junction1_det,junction2_det, FAZ, vessel = trans_tensor(image), trans_tensor(junction1_map),trans_tensor(junction2_map),\
                                trans_tensor(junction1_det),trans_tensor(junction2_det),trans_tensor(FAZ),trans_tensor(vessel)

        background_map = torch.ones_like(junction1_map) - junction1_map - junction2_map
        #imgBack = background.numpy()
        #cv2.imwrite('back.jpg', imgBack[0,:,:]*255)
        junction_map = torch.cat([junction1_map,junction2_map,background_map], 0)

        _,_, labelAll6, pointImg = genDetectLabel(junction1_det.squeeze().numpy(), \
                                                            junction2_det.squeeze().numpy(), self.girdSize)
        junction_det = labelAll6


        return image, junction_map,junction_det, FAZ, vessel
        
    def __len__(self):
        return len(self.imgPath)

    def getImgPath(self,root, isTraining):
        
        if isTraining:
            imgPath = os.path.join(root,'train', 'img')
        else:
            imgPath = os.path.join(root,'test', 'img')

        items = []
        imgList = os.listdir(imgPath)
        for name in imgList:
            pathRelative = os.path.join(imgPath,name)
            items.append(pathRelative)

        return items

    def getJuncMap(self,mat,shape):
        junc1 = np.zeros((shape,shape))
        junc2 = np.zeros((shape,shape))
        juncs = np.zeros((shape,shape))

        for point in mat['bifurcation']:
            junc1 = put_heatmap(junc1, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)

        for point in mat['crossing']:
            junc2 = put_heatmap(junc2, point, self.sigma)
            juncs = put_heatmap(juncs, point, self.sigma)
       
        #cv2.imwrite('junction_2.0.png',junc1*255)

        return Image.fromarray(junc1), Image.fromarray(junc2), Image.fromarray(juncs)

    def getFileName(self):
        return self.name


class datasetMT_drive(data.Dataset):
    #检测和heatmap的代码,drive数据集
    def __init__(self, root, task=['vessel','junctions'],sigma=2,size = 4, isTraining = True,imgSize = 512):
        #@task: could be vessel, junction, FAZ or All
        self.root = root
        self.taskName = task
        self.isTraining = isTraining
        self.name = ''
        self.sigma = sigma
        self.girdSize = size
        self.imgPath = self.getImgPath(root,isTraining)

    def __getitem__(self, index):
        imgPath = self.imgPath[index]
        image = Image.open(imgPath)
        image = image.convert('RGB')
        junction1 = []
        junction2 = []
        FAZ = []
        vessel = []
        #label
        gtPath = os.path.join(self.root, imgPath.split('/')[-3])
        self.name = imgPath.split('/')[-1]
        
        for task in self.taskName:
            
            if task == 'junctions':
                taskPath = os.path.join(gtPath, 'gt', task, self.name[:-4]+'_JunctionsPos.mat')
                juncMat = sio.loadmat(taskPath)
                junction1_map,junction2_map,_ = self.getJuncMap(juncMat,(512,512))

                junction1_det,junction2_det = drawPoint(juncMat,size=(512,512),index='DRIVE')
            else:
                taskPath = os.path.join(gtPath, 'gt', task, self.name[0:2]+'_manual1.gif')
                vessel = Image.open(taskPath)
                vessel = vessel.convert('L')

        image = image.resize((512,512))
        image = image.resize((512,512))
        vessel = vessel.resize((512,512))
        
        # augumentation,transform
        if self.isTraining:
            gamma_v = round(np.random.uniform(0.5,2.1),2)
            image = TF.adjust_gamma(img=image, gamma = gamma_v)

            rotate = 15
            angel = random.randint(-rotate, rotate)
            image = image.rotate(angel)

            junction1_map = junction1_map.rotate(angel)
            junction2_map = junction2_map.rotate(angel)

            junction1_det = junction1_det.rotate(angel)
            junction2_det = junction2_det.rotate(angel)

            vessel = vessel.rotate(angel)

            image, junction1_map,junction2_map, junction1_det,junction2_det, AV, vessel = \
                img_transforms_8(image, junction1_map,junction2_map, junction1_det,junction2_det, image, vessel, (512,512))

        else: 
            trans_tensor = transforms.ToTensor()

            image, junction1_map,junction2_map, junction1_det,junction2_det, AV, vessel = trans_tensor(image), trans_tensor(junction1_map),trans_tensor(junction2_map),\
                                trans_tensor(junction1_det),trans_tensor(junction2_det),trans_tensor(image),trans_tensor(vessel)

        
        # cv2.imwrite('a.jpg', Artery.numpy()*255)
        # cv2.imwrite('v.jpg', Vein.numpy()*255)
        # cv2.imwrite('u.jpg', unknow.numpy()*255)
        
        background_map = torch.ones_like(junction1_map) - junction1_map - junction2_map
        #imgBack = background.numpy()
        #cv2.imwrite('back.jpg', imgBack[0,:,:]*255)
        junction_map = torch.cat([junction1_map,junction2_map,background_map], 0)

        _,_, labelAll6, pointImg = genDetectLabel(junction1_det.squeeze().numpy(), \
                                                            junction2_det.squeeze().numpy(), self.girdSize)
        junction_det = labelAll6

        return (image,image,image), junction_map,junction_det, vessel, vessel
        
    def __len__(self):
        return len(self.imgPath)

    def getImgPath(self,root, isTraining):
        if isTraining:
            imgPath = os.path.join(root,'train', 'img')
        else:
            imgPath = os.path.join(root,'test', 'img')

        items = []
        imgList = os.listdir(imgPath)
        for name in imgList:
            pathRelative = os.path.join(imgPath,name)
            items.append(pathRelative)

        return items

    def getJuncMap(self,mat,shape):
        junc1 = np.zeros(shape)
        junc2 = np.zeros(shape)
        juncs = np.zeros(shape)

        for point in mat['BiffPos']:
            junc1 = put_heatmap(junc1, [(point[1]/565)*512,(point[0]/584)*512], self.sigma)
            juncs = put_heatmap(juncs, [(point[1]/565)*512,(point[0]/584)*512], self.sigma)

        for point in mat['CrossPos']:
            junc2 = put_heatmap(junc2, [(point[1]/565)*512,(point[0]/584)*512], self.sigma)
            juncs = put_heatmap(juncs, [(point[1]/565)*512,(point[0]/584)*512], self.sigma)
       
        #cv2.imwrite('junction_2.0.png',junc1*255)

        return Image.fromarray(junc1), Image.fromarray(junc2), Image.fromarray(juncs)

    def getFileName(self):
        return self.name


if __name__ == '__main__':
    # path = '/media/hjk/10E3196B10E3196B/dataSets/multi-task/DRIVE'
    # dataloader = DataLoader(datasetMT_drive(path), batch_size=1, num_workers=0, shuffle=True)
    # batch_iterator = iter(dataloader)
    # image, junction, FAZ, vessel = next(batch_iterator)
    value = 0.7
    path = '/media/hjk/10E3196B10E3196B/dataSets/Rose/train/img/01.png'
    image = Image.open(path)
    v = round(np.random.uniform(0.7,1.9),2)
    image = TF.adjust_gamma(img=image, gamma = value)
    image.save('07.png')
    
    print(round(v,2))
