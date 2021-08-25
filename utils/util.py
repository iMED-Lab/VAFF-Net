#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :util.py
@Description : some generic function
@Time        :2021/06/27 15:29:43
@Author      :Jinkui Hao
@Version     :1.0
'''
from evaluation import *
import cv2
import numpy as np
import os
import glob


def testSeg(pred,gt):
    #用于评价分割的结果，FAZ，vessel
    #输入是numpy
    AUC = 0
    mask_pred_ori = pred
    true_mask = gt
    #mask_pred_ori = mask_pred_ori[1,:,:]
    #true_mask = gt[1,:,:]
    th2 = ((mask_pred_ori) * 255).astype('uint8')
    #value, threshed_pred = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    value, threshed_pred = cv2.threshold(th2, 120, 255, cv2.THRESH_BINARY)


    mask_pred = torch.from_numpy(threshed_pred / 255.0)
    true_mask = torch.from_numpy(true_mask)
    DC = get_DC(mask_pred, true_mask)
    JS = jaccard_score(mask_pred, true_mask)
    
    temp = confusion(mask_pred, true_mask)
    Acc = temp["BAcc"]
    Sen = temp["TPR"]
    Spe = temp["TNR"]

    return AUC, DC, Acc, Sen, JS


if __name__ == '__main__':

    gtPath = '/home/imed/disk5TA/kevin/code/8.multi-task/ROSE-1_results/SVC_DVC/gt'
    resultPath = '/home/imed/disk5TA/kevin/code/8.multi-task/ROSE-1_results/SVC_DVC/methods'

    resList = os.listdir(resultPath)
    
    for method in resList:
        imgPath = os.path.join(resultPath,method)
        imgList = os.listdir(imgPath)
        DC_All = []
        Acc_All = []
        Sen_All = []
        JS_All = []
        for name in imgList:
            imgName = os.path.join(imgPath, name)
            img = cv2.imread(imgName,0)
            newName = glob.glob(os.path.join(gtPath,name[:40]+'*'))
            gt = cv2.imread(os.path.join(gtPath,newName[0]),0)
            AUC, DC, Acc, Sen, JS = testSeg(img/255,gt/255)
            DC_All.append(DC)
            Acc_All.append(Acc*100)
            Sen_All.append(Sen)
            JS_All.append(JS)

        DC_mean = np.mean(DC_All)
        Acc_mean = np.mean(Acc_All)
        Sen_mean = np.mean(Sen_All)
        JS_mean = np.mean(JS_All)

        DC_std = np.std(DC_All)
        Acc_std = np.std(Acc_All)
        Sen_std = np.std(Sen_All)
        JS_std = np.std(JS_All)

        # DC, Acc, JS:
        print(method,': %04f+%04f, %04f+%04f, %04f+%04f'%(DC_mean, DC_std, Acc_mean, Acc_std, JS_mean, JS_std))
        
    print('done~')