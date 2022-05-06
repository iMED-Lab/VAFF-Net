#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :config.py
@Description :configuration
@Time        :2020/10/30 14:45:22
@Author      :Jinkui Hao
@Version     :1.0
'''

import os 

class Config3In():
    dataPath = 'data/DRIVE-MT'
    resultPath = 'result/multi-task' #path for saving model and results

    saveName = 'MT_DRIVE-ours' 
    savePath = os.path.join(resultPath, saveName)
    env = saveName #visdom

    imgSize = 512

    #train
    batch_size = 1
    num_epochs = 1000
    base_lr = 5e-5
    weight_decay = 5e-5

    #dataset
    taskName = 'MT'  # vessel, junctions, FAZ
    sigma = 3  #for guass point of junction

    gpu = '3'

    isDetectModel = True
    is2Channel = True
    gridSize = 8 
    confidence = 0.5
    tolerance = 9

    lossWeight = [4,1,2]  #onj, no_obj, class
    weight = 'dwa'
    temp = 2
