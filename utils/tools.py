#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :utli.py
@Description :
@Time        :2020/10/31 10:15:11
@Author      :Jinkui Hao
@Version     :1.0
'''
import cv2
import os
import math
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from PIL import Image,ImageDraw
import scipy.io as sio


def singleDet(pred, gt, tolerance=7,size=(304,304)):
    #只检测一个通道的准确率
    epsilon = 1e-7
    _, gt = cv2.threshold(gt, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, pred = cv2.threshold(pred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #根据容忍度计算一幅图像中总的样本数量
    gt_total_num = int(size[0]/tolerance)**2

    #统计gt，pred的总数
    pred_p_num = np.sum(pred)
    gt_p_num = np.sum(gt)
    #print('predicted and gt numbers %d, %d:'%(pred_p_num,gt_p_num))

    #腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tolerance, tolerance))
    gt_dilate = cv2.dilate(gt, kernel)
    # cv2.imwrite('gtDilte-5.jpg',gt_dilate*255)
    # cv2.imwrite('pred-5.jpg',pred*255)

    orResult = cv2.bitwise_or(gt_dilate, pred)

    #false positive and TP
    FP = np.sum(orResult)-np.sum(gt_dilate)
    FP = FP if FP > 0 else 0

    TP = pred_p_num - FP
    TP = TP if TP > 0 else 0

    if gt_p_num < TP:
        FN = 0
    else:
        FN = gt_p_num - TP
    #FN = FN if FN > 0 else 0

    TN = gt_total_num - gt_p_num -FP

    TN = TN if TN > 0 else 0

    #print('FP, TP, FN, TN: %d, %d, %d, %d'%(FP,TP,FN,TN))

    Sen = TP/(TP+FN+epsilon)
    Spe = TN/(TN+FP+epsilon)
    Pre = TP/(TP+FP+epsilon)
    F1 = (2*Pre*Sen)/(Pre+Sen+epsilon)

    #print('Sen, Spe, Pre, F1: %04f, %04f, %04f, %04f'%(Sen, Spe, Pre, F1))

    #cv2.imwrite('orres.jpg',orResult*255)


    return Sen, Spe, Pre, F1, gt_p_num

def classifyMatrixV3(BF_pred, CR_pred, BF_gt, CR_gt, tolerance=7,size=(304,304)):
    #根据gt和pred的点图像生成分类的评价矩阵，返回评价结果
    #思路:
    #将所有的pred样本作为总体，BF和CR分开评价，统计BF的时候，CR和错误的作为负类
    # 1.先得到所有junction的TP，基于TP做分类的评价
    # 2. 取GT对应位置的值，统计TP等信息

    #这里输入的格式应该是两通道的预测的点，分别是BF 和 CR
    epsilon = 1e-7

    _, BF_pred = cv2.threshold(BF_pred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, CR_pred = cv2.threshold(CR_pred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, BF_gt = cv2.threshold(BF_gt, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, CR_gt = cv2.threshold(CR_gt, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gt = cv2.bitwise_or(BF_gt, CR_gt)
    pred = cv2.bitwise_or(BF_pred, CR_pred)
    # cv2.imwrite('gt.jpg',gt*255)
    # cv2.imwrite('pred.jpg',pred*255)

    #先统计TP，得到相应的mask
   
    
    #根据容忍度计算一幅图像中总的样本数量
    gt_total_num = int(size[0]/tolerance)**2

    #统计gt，pred的总数
    pred_p_num = np.sum(pred)
    gt_p_num = np.sum(gt)
    #print('predicted and gt numbers %d, %d:'%(pred_p_num,gt_p_num))

    #膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tolerance, tolerance))
    gt_dilate = cv2.dilate(gt, kernel)
    cv2.imwrite('gtDilte-5.jpg',gt_dilate*255)
    cv2.imwrite('pred-5.jpg',pred*255)

    #orResult = cv2.bitwise_and(gt_dilate, pred)
    
    #获取TP的mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tolerance, tolerance))
    all_pred_dilated = cv2.dilate(pred, kernel)
    cv2.imwrite('all_pred_dilated.jpg',all_pred_dilated*255)

    #pred_point,里面包含了2类,1代表BF，2代表CR
    # pred_point = cv2.dilate(cv2.bitwise_and(all_pred_dilated, BF_pred),kernel) + \
    #                  cv2.dilate(cv2.bitwise_and(all_pred_dilated, CR_pred),kernel)*2

    pred_point = cv2.bitwise_and(all_pred_dilated, BF_pred) + cv2.bitwise_and(all_pred_dilated, CR_pred)*2
    #gt_point = cv2.bitwise_and(TP_mask, BF_gt) + cv2.bitwise_and(TP_mask, CR_gt)*2

    #计算总样本数量
    num_total = np.sum(pred)

    Sen1, Spe1, Pre1, F11, num_BF = singleDet(BF_pred, BF_gt, tolerance=tolerance,size = size)
    Sen2, Spe2, Pre2, F12, num_CR = singleDet(CR_pred, CR_gt, tolerance=tolerance,size = size)
    total = num_CR+num_BF

    BF_gt_dilate = cv2.dilate(BF_gt, kernel)
    CR_gt_dilate = cv2.dilate(CR_gt, kernel)

     #计算BF的TP值
    TP_BF = np.sum((pred_point == 1) & (BF_gt_dilate == 1))
    FP_BF = np.sum((pred_point == 1) & (BF_gt_dilate == 0))
    TN_BF = np.sum((pred_point == 2) & (BF_gt_dilate == 0))
    FN_BF = np.sum((pred_point == 2) & (BF_gt_dilate == 1))
    # print('FP_BF, TP_BF, FN_BF, TN_BF: %d, %d, %d, %d'%(FP_BF,TP_BF,FN_BF,TN_BF))


    Sen_BF = TP_BF/(TP_BF+FN_BF+epsilon)
    Spe_BF = TN_BF/(TN_BF+FP_BF+epsilon)
    Pre_BF = TP_BF/(TP_BF+FP_BF+epsilon)
    F1_BF = (2*Pre_BF*Sen_BF)/(Pre_BF+Sen_BF+epsilon)
    

    #CR
    TP_CR = np.sum((pred_point == 2) & (CR_gt_dilate == 1))
    FP_CR = np.sum((pred_point == 2) & (CR_gt_dilate == 0))
    TN_CR = np.sum((pred_point == 1) & (CR_gt_dilate == 0))
    FN_CR = np.sum((pred_point == 1) & (CR_gt_dilate == 1))
    # print('FP_CR, TP_CR, FN_CR, TN_CR: %d, %d, %d, %d'%(FP_CR,TP_CR,FN_CR,TN_CR))

    Sen_CR = TP_CR/(TP_CR+FN_CR+epsilon)
    Spe_CR = TN_CR/(TN_CR+FP_CR+epsilon)
    Pre_CR = TP_CR/(TP_CR+FP_CR+epsilon)
    F1_CR = (2*Pre_CR*Sen_CR)/(Pre_CR+Sen_CR+epsilon)

    # Sen, Spe, Pre, F1 = (Sen_BF+Sen_CR)/(2),(Spe_BF+Spe_CR)/2, (Pre_BF+Pre_CR)/2, (F1_BF+F1_CR)/2

    Sen, Spe, Pre, F1 = (num_BF*Sen_BF+Sen_CR*num_CR)/(total), (num_BF*Spe_BF+Spe_CR*num_CR)/(total), \
                        (num_BF*Pre_BF+Pre_CR*num_CR)/(total), (num_BF*F1_BF+F1_CR*num_CR)/(total)

    return Sen, Spe, Pre, F1

def detectionMatrix(BF_pred, CR_pred, BF_gt, CR_gt, tolerance=7,size=(304,304)):
    #根据gt和pred的点图像生成评价矩阵，返回评价结果
    #接受单通道的图像，这里评价的是检测的表现，不涉及分类准确率
    #思路:先对GT进行膨胀，得到容忍后的gt,
    #2.将pred和原图对比，计算TP,FP等

    gt = cv2.bitwise_or(BF_gt, CR_gt)
    pred = cv2.bitwise_or(BF_pred, CR_pred)

    epsilon = 1e-7
    _, gt = cv2.threshold(gt, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, pred = cv2.threshold(pred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #根据容忍度计算一幅图像中总的样本数量
    gt_total_num = int(size[0]/tolerance)**2

    #统计gt，pred的总数
    pred_p_num = np.sum(pred)
    gt_p_num = np.sum(gt)
    #print('predicted and gt numbers %d, %d:'%(pred_p_num,gt_p_num))

    #腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tolerance, tolerance))
    gt_dilate = cv2.dilate(gt, kernel)
    # cv2.imwrite('gtDilte-5.jpg',gt_dilate*255)
    # cv2.imwrite('pred-5.jpg',pred*255)

    orResult = cv2.bitwise_or(gt_dilate, pred)

    #false positive and TP
    FP = np.sum(orResult)-np.sum(gt_dilate)
    FP = FP if FP > 0 else 0

    TP = pred_p_num - FP
    TP = TP if TP > 0 else 0

    if gt_p_num < TP:
        FN = 0
    else:
        FN = gt_p_num - TP
    #FN = FN if FN > 0 else 0

    TN = gt_total_num - gt_p_num -FP

    TN = TN if TN > 0 else 0

    #print('FP, TP, FN, TN: %d, %d, %d, %d'%(FP,TP,FN,TN))

    Sen = TP/(TP+FN+epsilon)
    Spe = TN/(TN+FP+epsilon)
    Pre = TP/(TP+FP+epsilon)
    F1 = (2*Pre*Sen)/(Pre+Sen+epsilon)

    #print('Sen, Spe, Pre, F1: %04f, %04f, %04f, %04f'%(Sen, Spe, Pre, F1))

    #cv2.imwrite('orres.jpg',orResult*255)


    return Sen, Spe, Pre, F1

def detectionMatrix_one(pred, gt, tolerance=7,size=(304,304)):
    #根据gt和pred的点图像生成评价矩阵，返回评价结果
    #合并
    #接受单通道的图像，这里评价的是检测的表现，不涉及分类准确率
    #思路:先对GT进行膨胀，得到容忍后的gt,
    #2.将pred和原图对比，计算TP,FP等

    # gt = cv2.bitwise_or(BF_gt, CR_gt)
    # pred = cv2.bitwise_or(BF_pred, CR_pred)

    epsilon = 1e-7
    _, gt = cv2.threshold(gt, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, pred = cv2.threshold(pred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #根据容忍度计算一幅图像中总的样本数量
    gt_total_num = int(size[0]/tolerance)**2

    #统计gt，pred的总数
    pred_p_num = np.sum(pred)
    gt_p_num = np.sum(gt)
    #print('predicted and gt numbers %d, %d:'%(pred_p_num,gt_p_num))

    #腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tolerance, tolerance))
    gt_dilate = cv2.dilate(gt, kernel)
    # cv2.imwrite('gtDilte-5.jpg',gt_dilate*255)
    # cv2.imwrite('pred-5.jpg',pred*255)

    orResult = cv2.bitwise_or(gt_dilate, pred)

    #false positive and TP
    FP = np.sum(orResult)-np.sum(gt_dilate)
    FP = FP if FP > 0 else 0

    TP = pred_p_num - FP
    TP = TP if TP > 0 else 0

    if gt_p_num < TP:
        FN = 0
    else:
        FN = gt_p_num - TP
    #FN = FN if FN > 0 else 0

    TN = gt_total_num - gt_p_num -FP

    TN = TN if TN > 0 else 0

    #print('FP, TP, FN, TN: %d, %d, %d, %d'%(FP,TP,FN,TN))

    Sen = TP/(TP+FN+epsilon)
    Spe = TN/(TN+FP+epsilon)
    Pre = TP/(TP+FP+epsilon)
    F1 = (2*Pre*Sen)/(Pre+Sen+epsilon)

    #print('Sen, Spe, Pre, F1: %04f, %04f, %04f, %04f'%(Sen, Spe, Pre, F1))

    #cv2.imwrite('orres.jpg',orResult*255)
    return Sen, Spe, Pre, F1

def drawPoint_point(predict,isTwo = False, cir_color = 'yellow',radio = 2,size=(304,304)):
    #将heatmap求局部最大值，将其画成单个关键点
    #返回Image格式的图像
    #@isTwo if . is True, 同时返回有圆圈的图，格式是PIL 的Image
    #@radio: 半径
    empty_array = np.zeros(size)
    empty_array = Image.fromarray(empty_array)

    if empty_array.mode != "RGB":
        empty_array = empty_array.convert("RGB")

    color = (255,255,255)
    draw = ImageDraw.Draw(empty_array)

    empty_array_cir = np.zeros(size)
    empty_array_cir = Image.fromarray(empty_array_cir)

    if empty_array_cir.mode != "RGB":
        empty_array_cir = empty_array_cir.convert("RGB")

    color = (255,255,255)
    draw = ImageDraw.Draw(empty_array)
    draw_cir = ImageDraw.Draw(empty_array_cir)

    ori_bif,ori_cross = 0,0
    
    before = np.uint8(predict)
    #自动阈值
    value, predict = cv2.threshold(predict, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    predict = predict*before
    coordinates = peak_local_max(predict, min_distance=2)  #返回[行，列]，即[y, x]
    predict = Image.fromarray(predict)
    predict = predict.convert('RGB')

    for i in range(coordinates.shape[0]):
        x,y = coordinates[i,0], coordinates[i,1]
        draw.point((int(y), int(x)),fill=(255,255,255))
        #draw_cir.point((int(y), int(x)),fill=(255,255,255))
        draw_cir.ellipse((int(y)-radio, int(x)-radio, int(y)+radio, int(x)+radio), outline=cir_color, width=1)

    empty_array = empty_array.convert('L')
    empty_array = np.array(empty_array)

    if isTwo:
        return empty_array.astype(np.uint8), empty_array_cir
    else:
        return empty_array.astype(np.uint8)

def convertPoint(gt,pred):
    #将heatmap的点转化为单独的点返回
    #输入参数是直接来自heatmap的点
    size = gt[0,:,:].shape
    BF_gt = np.uint8(np.zeros(gt[0,:,:].shape))
    maxIndex = np.argmax(gt,axis=0)
    BF_ori = np.uint8(gt[0,:,:])
    BF_gt[maxIndex == 0] = BF_ori[maxIndex == 0]
    cv2.imwrite('bf-gt.jpg',BF_gt)

    CR_gt = np.uint8(np.zeros(gt[0,:,:].shape))
    CR_ori = np.uint8(gt[1,:,:])
    CR_gt[maxIndex == 1] = CR_ori[maxIndex == 1]

    BF_pred = np.uint8(np.zeros(pred[0,:,:].shape))
    maxIndex = np.argmax(pred,axis=0)
    BF_ori = np.uint8(pred[0,:,:])
    BF_pred[maxIndex == 0] = BF_ori[maxIndex == 0]
    #cv2.imwrite('bf-gt.jpg',BF_pred)

    CR_pred = np.uint8(np.zeros(pred[0,:,:].shape))
    CR_ori = np.uint8(pred[1,:,:])
    CR_pred[maxIndex == 1] = CR_ori[maxIndex == 1]

    BF_gt,BF_gt_c = drawPoint_point(BF_gt,isTwo=True,size=size)
    CR_gt,CR_gt_c = drawPoint_point(CR_gt,isTwo=True,cir_color='green',size=size)
    BF_pred,BF_pred_c = drawPoint_point(BF_pred,isTwo=True,size=size)
    CR_pred,CR_pred_c = drawPoint_point(CR_pred,isTwo=True,cir_color='green',size=size)

    return BF_pred, CR_pred, BF_gt, CR_gt,BF_pred_c, CR_pred_c, BF_gt_c, CR_gt_c

def heatmap2point(gt,size=(304,304)):
    #将heatmap的点转化为单独的点返回
    #输入参数是直接来自heatmap的点

    BF_gt = np.uint8(np.zeros(gt[0,:,:].shape))
    maxIndex = np.argmax(gt,axis=0)
    BF_ori = np.uint8(gt[0,:,:])
    BF_gt[maxIndex == 0] = BF_ori[maxIndex == 0]
    #cv2.imwrite('bf-gt.jpg',BF_gt)

    CR_gt = np.uint8(np.zeros(gt[0,:,:].shape))
    CR_ori = np.uint8(gt[1,:,:])
    CR_gt[maxIndex == 1] = CR_ori[maxIndex == 1]

    BF_gt, BF_gt_cir = drawPoint_point(BF_gt,isTwo=True,cir_color='yellow',size=size)
    CR_gt, CR_gt_cir = drawPoint_point(CR_gt,isTwo=True,cir_color='green',size=size)

    return BF_gt, CR_gt,BF_gt_cir,CR_gt_cir

def convertPoint_one(gt,pred,size=(304,304)):
    #将heatmap的点转化为单独的点返回
    #heatmap为单通道
    #输入参数是直接来自heatmap的点

    all_gt = np.uint8(gt)
    #cv2.imwrite('all-gt.jpg',all_gt)

    all_pred = np.uint8(pred)
    #cv2.imwrite('all-gt.jpg',all_pred)

    all_gt = drawPoint_point(all_gt,size=size)
    all_pred = drawPoint_point(all_pred,size=size)

    return all_pred,all_gt

def drawPoint(juncMat,size=(304,304),index = 'ROSE'):
    #根据 .mat文件生成图片，非heatmap,单个点
    junc1 = np.zeros(size)
    junc2 = np.zeros(size)

    junc1 = Image.fromarray(junc1)
    junc2 = Image.fromarray(junc2)

    # Convert to RGB mode
    if junc1.mode != "RGB":
        junc1 = junc1.convert("RGB")

    if junc2.mode != "RGB":
        junc2 = junc2.convert("RGB")

    color = (255,255,255)
    draw1 = ImageDraw.Draw(junc1)
    draw2 = ImageDraw.Draw(junc2)

    if index == 'ROSE':
        for point in juncMat['bifurcation']:
            center_x, center_y = point
            draw1.point((int(center_x), int(center_y)),fill=(255,255,255))

        for point in juncMat['crossing']:
            center_x, center_y = point
            draw2.point((int(center_x), int(center_y)),fill=(255,255,255))
    else:
        for point in juncMat['BiffPos']:
            center_x, center_y = point
            draw1.point((int(center_y), int(center_x)),fill=(255,255,255))

        for point in juncMat['CrossPos']:
            center_x, center_y = point
            draw2.point((int(center_y), int(center_x)),fill=(255,255,255))

    return junc1.convert('L'), junc2.convert('L')

def imgTrans():
    #图像格式转换
    path = '/media/hjk/10E3196B10E3196B/dataSets/Rose/test/gt/vessel'
    imgList = os.listdir(path)
    for name in imgList:
        image = cv2.imread(os.path.join(path,name))
        cv2.imwrite(os.path.join(path,name[:2]+'.png'),image)

def put_heatmap(heatmap, center, sigma):
    """
    Parameters
    -heatmap: 热图（heatmap）
    - center： 关键点的位置
    - sigma: 生成高斯分布概率时的一个参数
    Returns
    - heatmap: 热图
    """
 
    center_x, center_y = center
    height, width = heatmap.shape
 
    th = 4.6052
    delta = math.sqrt(th * 2)
 
    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))
 
    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))
 
    exp_factor = 1 / 2.0 / sigma / sigma
 
    ## fast - vectorize
    arr_heatmap = heatmap[y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y)**2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)

    return heatmap

def showPoint(image, predict):
    #可视化预测的点,from heatmap
    ori_bif,ori_cross = 0,0
    predicts = []
    if predict.shape[0] == 3:
        #由于用了softmax，这里先根据最大值来分别得到bifur,crossing
        burfaction = np.uint8(np.zeros(predict[0,:,:].shape))

        maxIndex = np.argmax(predict,axis=0)
        burfaction_ori = np.uint8(predict[0,:,:])
        burfaction[maxIndex == 0] = burfaction_ori[maxIndex == 0]

        ori_bif = burfaction
        
        value, burfaction = cv2.threshold(burfaction, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #value, burfaction = cv2.threshold(burfaction, 128, 255, cv2.THRESH_BINARY)
        #距离变换
        # burfaction = cv2.distanceTransform(burfaction, cv2.DIST_L1, cv2.DIST_MASK_3)
        # value, burfaction = cv2.threshold(cv2.equalizeHist(cv2.convertScaleAbs(burfaction)), 220, 255, cv2.THRESH_BINARY)
        burfaction = burfaction*ori_bif

        coordinates = peak_local_max(burfaction, min_distance=2)  #返回[行，列]，即[y, x]
        burfaction = Image.fromarray(burfaction)
        burfaction = burfaction.convert('RGB')
        burfaction = np.asarray(burfaction)
        #value, burfaction = cv2.threshold(burfaction, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for i in range(coordinates.shape[0]):
            x,y = coordinates[i,0], coordinates[i,1]
            cv2.circle(image,(y,x),3, (0,255,255),0)
            
            cv2.circle(burfaction,(y,x),3, (0,255,255),0)
        predicts.append(burfaction)

        crossing = np.uint8(np.zeros(predict[0,:,:].shape))
        crossing_ori = np.uint8(predict[1,:,:])
        crossing[maxIndex == 1] = crossing_ori[maxIndex == 1]
        #crossing = np.uint8(predict[1,:,:])
        ori_cross = crossing
        value, crossing = cv2.threshold(np.uint8(crossing), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #value, crossing = cv2.threshold(np.uint8(crossing), 128, 255, cv2.THRESH_BINARY)
        #距离变换
        # crossing = cv2.distanceTransform(crossing, cv2.DIST_L1, cv2.DIST_MASK_3)
        # value, crossing = cv2.threshold(cv2.equalizeHist(cv2.convertScaleAbs(crossing)), 220, 255, cv2.THRESH_BINARY)
        crossing = crossing*ori_cross
        coordinates = peak_local_max(crossing, min_distance=2)  #返回[行，列]，即[y, x]
        crossing = Image.fromarray(crossing)
        crossing = crossing.convert('RGB')
        crossing = np.asarray(crossing)*255
        for i in range(coordinates.shape[0]):
            x,y = coordinates[i,0], coordinates[i,1]
            cv2.circle(image,(y,x),3, (0,255,0),0)
            
            cv2.circle(crossing,(y,x),3, (0,255,0),0)
        predicts.append(crossing)

    else:
        before = np.uint8(predict)
        #自动阈值
        value, predict = cv2.threshold(predict, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #value, predict = cv2.threshold(predict, 50, 255, cv2.THRESH_BINARY)
        #cv2.imwrite('pred_1.jpg',predict)
        #距离变换
        # predict = cv2.distanceTransform(predict, cv2.DIST_L1, cv2.DIST_MASK_3)
        # value, predict = cv2.threshold(cv2.equalizeHist(cv2.convertScaleAbs(predict)), 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        predict = predict*before
        coordinates = peak_local_max(predict, min_distance=2)  #返回[行，列]，即[y, x]
        #image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        predict = Image.fromarray(predict)
        predict = predict.convert('RGB')
        predict = np.asarray(predict)
        for i in range(coordinates.shape[0]):
            x,y = coordinates[i,0], coordinates[i,1]
            cv2.circle(image,(y,x),3, (0,255,255),0)
            cv2.circle(predict,(y,x),3, (0,255,0),0)
        predicts.append(predict)
    #cv2.imwrite('result.jpg',predict)
    return image, predicts, ori_bif,ori_cross

def genDetectLabel(junc1, junc2, size=8):
    
    #根据图片label生成用于检测的label:
    #对于每个region, coor:[x,y,c]  category:[0,0,1]（background, bifurcation, crossing)
    #@juncMat: 存放坐标信息的.mat矩阵
    junc1, junc2 = junc1*255, junc2*255
    visImg = np.zeros(junc1.shape) 
    value, junc1 = cv2.threshold(np.uint8(junc1), 220, 1, cv2.THRESH_BINARY)
    value, junc2 = cv2.threshold(np.uint8(junc2), 220, 1, cv2.THRESH_BINARY)
    # cv2.imwrite('1.jpg', junc1*255)
    # cv2.imwrite('2.jpg', junc2*255)
    gridSize = size
    gridNum = int(junc1.shape[0]/gridSize)

    labelCood = np.zeros((gridNum,gridNum,3))
    labelCate = np.zeros((gridNum,gridNum,3))
    labelAll6 = np.zeros((gridNum,gridNum,6))

    for i in range(gridNum):
        for j in range(gridNum):
            grid1 = junc1[i*gridSize:(i+1)*gridSize, j*gridSize:(j+1)*gridSize]
            grid2 = junc2[i*gridSize:(i+1)*gridSize, j*gridSize:(j+1)*gridSize]
            tempImg = np.zeros(junc1.shape)
            if np.sum(grid2) > 0:
                #crossing
                labelCate[i,j,2] = 1
                labelAll6[i,j,5] = 1

                tempImg[i*gridSize:(i+1)*gridSize, j*gridSize:(j+1)*gridSize] = grid2
                visImg[i*gridSize:(i+1)*gridSize, j*gridSize:(j+1)*gridSize] = 255
                coor = np.where(tempImg == 1)
                coorX,coorY = coor[0][0], coor[1][0]
                labelCood[i,j,0] = coorX/junc1.shape[0]
                labelCood[i,j,1] = coorY/junc1.shape[0]
                labelCood[i,j,2] = 1

                labelAll6[i,j,0] = coorX/junc1.shape[0]
                labelAll6[i,j,1] = coorY/junc1.shape[0]
                labelAll6[i,j,2] = 1

            elif np.sum(grid1) > 0:
                #bifurcation
                labelCate[i,j,1] = 1
                labelAll6[i,j,4] = 1

                tempImg[i*gridSize:(i+1)*gridSize, j*gridSize:(j+1)*gridSize] = grid1
                visImg[i*gridSize:(i+1)*gridSize, j*gridSize:(j+1)*gridSize] = 100
                coor = np.where(tempImg == 1)
                coorX,coorY = coor[0][0], coor[1][0]
                labelCood[i,j,0] = coorX/junc1.shape[0]
                labelCood[i,j,1] = coorY/junc1.shape[0]
                labelCood[i,j,2] = 1

                labelAll6[i,j,0] = coorX/junc1.shape[0]
                labelAll6[i,j,1] = coorY/junc1.shape[0]
                labelAll6[i,j,2] = 1
            else:
                labelCate[i,j,0] = 1
                labelAll6[i,j,3] = 1
            #cv2.imwrite('resVisulize.jpg', visImg+junc1*255)

    return labelCood, labelCate, labelAll6, np.uint8(visImg+junc1*255)

def showPrediction(prediction,thresh=0.5,gridSize=4,isGT = True,size = (304,304)):
    #将检测的label进行可视化,基于检测的结果
    #prediction should be a numpy array, and its shape should be [s,s,6], s is grid number.

    #draw point
    resImage = np.zeros(size)
    resImage = Image.fromarray(resImage)
    resImage = resImage.convert('RGB')

    #只画点，不画区域-for BF
    pointImage_BF = np.zeros(size)
    pointImage_BF = Image.fromarray(pointImage_BF)
    pointImage_BF = pointImage_BF.convert('RGB')

    #只画点，不画区域-for CR
    pointImage_CR = np.zeros(size)
    pointImage_CR = Image.fromarray(pointImage_CR)
    pointImage_CR = pointImage_CR.convert('RGB')

    color = (255,255,255)
    draw = ImageDraw.Draw(resImage)
    drawPoint_BF = ImageDraw.Draw(pointImage_BF)
    drawPoint_CR = ImageDraw.Draw(pointImage_CR)

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            x,y = prediction[i,j,0], prediction[i,j,1]
            c = prediction[i,j,2]
            catePred = prediction[i,j,3:]
            category = np.argmax(catePred)
            if c > thresh:
                # color = (0,255,0)
                # draw.rectangle((j*gridSize,i*gridSize-1,(j+1)*gridSize,(i+1)*gridSize-1), fill=color)
                # draw.point((int(y*304), int(x*304)),fill=(255,255,255))
                if category == 1:
                    color = (200,20,0)
                    draw.rectangle((j*gridSize,i*gridSize-1,(j+1)*gridSize,(i+1)*gridSize-1), fill=color)
                    #draw.point((int(y*304),int(x*304),),fill=(255,255,255))
                    if isGT:
                        drawPoint_BF.point((int(y*size[1]),int(x*size[0])),fill=(255,255,255))
                    else:
                        drawPoint_BF.point((j*gridSize+1,i*gridSize+1),fill=(255,255,255))

                elif category == 2:
                    color = (0,20,200)
                    draw.rectangle((j*gridSize,i*gridSize-1,(j+1)*gridSize,(i+1)*gridSize-1), fill=color)
                    #draw.point((int(y*304), int(x*304)),fill=(255,255,255))
                    if isGT:
                        drawPoint_CR.point((int(y*size[1]),int(x*size[0])),fill=(255,255,255))
                    else:
                        drawPoint_CR.point((j*gridSize+1,i*gridSize+1),fill=(255,255,255))

    #resImage.save('result.jpg')
    return resImage, pointImage_BF, pointImage_CR

 #heatmap和region的组合，生成结果

def getCMBResult(prediction_det,prediction_map,thresh=0.5,gridSize=4,isGT = True,size=(304,304)):
    #将检测的label进行可视化,基于检测和heatmap的结果
    #prediction should be a numpy array, and its shape should be [s,s,6], s is grid number.

    #只画点，不画区域-for BF
    pointImage_BF = np.zeros(size)
    pointImage_BF = Image.fromarray(pointImage_BF)
    pointImage_BF = pointImage_BF.convert('RGB')

    #只画点，不画区域-for CR
    pointImage_CR = np.zeros(size)
    pointImage_CR = Image.fromarray(pointImage_CR)
    pointImage_CR = pointImage_CR.convert('RGB')

    color = (255,255,255)
    drawPoint_BF = ImageDraw.Draw(pointImage_BF)
    drawPoint_CR = ImageDraw.Draw(pointImage_CR)

    for i in range(prediction_det.shape[0]):
        for j in range(prediction_det.shape[1]):
            x,y = prediction_det[i,j,0], prediction_det[i,j,1]
            c = prediction_det[i,j,2]
            catePred = prediction_det[i,j,3:]
            category = np.argmax(catePred)
            if c > thresh:
                # draw.rectangle((j*gridSize,i*gridSize-1,(j+1)*gridSize,(i+1)*gridSize-1), fill=color)
                # draw.point((int(y*304), int(x*304)),fill=(255,255,255))
                if category == 1:
                    drawPoint_BF.rectangle((j*gridSize,i*gridSize-1,(j+1)*gridSize,(i+1)*gridSize-1), fill=color)                

                elif category == 2:
                    drawPoint_CR.rectangle((j*gridSize,i*gridSize-1,(j+1)*gridSize,(i+1)*gridSize-1), fill=color)
    BF = pointImage_BF.convert('L')
    CR = pointImage_CR.convert('L')
    BF = np.asarray(BF)
    CR = np.asarray(CR)
    value, BF = cv2.threshold(BF, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    value, CR = cv2.threshold(CR, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    map = np.uint8(prediction_map)

    CR = map*CR
    BF = map*BF

    # cv2.imwrite('CR-after.jpg',CR)
    # cv2.imwrite('BF-after.jpg',BF)

    BF_pred, BF_cir = drawPoint_point(BF,isTwo=True,cir_color='yellow',size=size)
    CR_pred, CR_cir = drawPoint_point(CR,isTwo=True,cir_color='green',size=size)

    # BF_cir.save('fff.jpg')
    # CR_cir.save('fff-2.jpg')

    # cv2.imwrite('CR-after-2.jpg',CR_pred)
    # cv2.imwrite('BF-after-2.jpg',BF_pred)

    #resImage.save('result.jpg')
    return map, BF_pred, CR_pred, BF_cir, CR_cir

if __name__ == '__main__':
    #imgTrans()
    # taskPath = '/media/hjk/10E3196B10E3196B/dataSets/Rose/test/gt/junctions/02.mat'
    # juncMat = sio.loadmat(taskPath)
    # junc1, junc2 = drawPoint(juncMat)
    # junc1.save('data/bifurcation.jpg')
    # junc2.save('data/crossing.jpg')
    
    # #根据heatmap画单个关键点
    # juncImg = cv2.imread('data/pred_cross_02.png', 0)
    # image = drawPoint_point(juncImg)
    # image.save('data/pred_CR_02.jpg')
    
    #根据gt和预测的关键点计算metrix
    BF_gt = cv2.imread('data/bifurcation_02.jpg',0)
    BF_pred = cv2.imread('data/pred_BF_02.jpg',0)

    CR_gt = cv2.imread('data/crossing_02.jpg',0)
    CR_pred = cv2.imread('data/pred_CR_02.jpg',0)

    Sen, Spe, Pre, F1 = classifyMatrix(BF_pred, CR_pred, BF_gt, CR_gt, tolerance=7)
    print('Sen, Spe, Pre, F1: %04f, %04f, %04f, %04f'%(Sen, Spe, Pre, F1))
    Sen, Spe, Pre, F1 = detectionMatrix(BF_pred, CR_pred, BF_gt, CR_gt, tolerance=7)
    print('Sen, Spe, Pre, F1: %04f, %04f, %04f, %04f'%(Sen, Spe, Pre, F1))
    # evaulatMatrix(gt, pred)
    print('done~')


    