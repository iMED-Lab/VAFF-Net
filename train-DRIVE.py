#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :train.py
@Description :Training and test of our VAFF-Net
@Time        :2021/06/30 14:45:15
@Author      :Jinkui Hao
@Version     :2.0
'''
import os
import sys
from torch.utils.data import DataLoader
from torch import optim, nn
import torch
from utils.Visualizer import Visualizer
from  models.ourModel import *
import numpy as np
import csv
from  datasets import datasetMT_drive
import random
from config import Config3In
from utils.WarmUpLR import WarmupLR
from evaluation.evaluation import *
from utils.tools import *
from utils.dice_loss import *
from utils.detectLoss import junctionLoss
from torchvision import transforms

# set seed
GLOBAL_SEED = 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def get_lr(optier):
    for param_group in optier.param_groups:
        return param_group['lr']


def testSeg(pred,gt,imgName,image,name='vessel'):
    AUC = 0
    mask_pred_ori = pred
    true_mask = gt
    th2 = ((mask_pred_ori) * 255).astype('uint8')
    value, threshed_pred = cv2.threshold(th2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pathJ = os.path.join(save_dir,name)
    if not os.path.isdir(os.path.join(pathJ,'Thresh')):
        os.makedirs(os.path.join(pathJ,'Thresh'))
    if not os.path.isdir(os.path.join(pathJ,'noThresh')):
        os.makedirs(os.path.join(pathJ,'noThresh'))

    colorFAZ = np.zeros((image.shape))
    colorFAZ[:,:,1] = threshed_pred
    CR_cir = cv2.cvtColor(np.asarray(threshed_pred),cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(pathJ,'Thresh',imgName[:-4]+'_ori.jpg'),0.8*image+0.2*colorFAZ)
    cv2.imwrite(os.path.join(pathJ,'Thresh',imgName[:-4]+'.jpg'),threshed_pred)
    cv2.imwrite(os.path.join(pathJ,'noThresh',imgName[:-4]+'.jpg'),th2)

    mask_pred = torch.from_numpy(threshed_pred / 255.0)
    true_mask = torch.from_numpy(true_mask)
    DC = get_DC(mask_pred, true_mask)
    JS = jaccard_score1(mask_pred, true_mask)
    
    
    temp = confusion(mask_pred, true_mask)
    Acc = temp["BAcc"]
    Sen = temp["TPR"]
    Spe = temp["TNR"]

    return AUC, DC, Acc, Sen, JS


def test_(model,dataloader,task, isSave = True):
    
    funcMtx_Vess = {'AUC':[], 'DC':[], 'Acc':[], 'Sen':[], 'JS':[]}

    funcMtx_cmb = {'SEN_det':[], 'SPE_det':[], 'PRE_det':[], 'F1_det':[], 
                    'SEN_cla':[], 'SPE_cla':[], 'PRE_cla':[], 'F1_cla':[]}

    i =0
    with torch.no_grad():
         for image, junction_map, junction_det, FAZ, vessels in dataloader:
            print('Evaluate %03d...' %i)
            i += 1
            
            input_1, input_2, input_3 = image
            input_1 = input_1.to(device)
            input_2 = input_2.to(device)
            input_3 = input_3.to(device)

            vessel_w = vessels


            gt_det = junction_det
            labels_det = gt_det.to(device)

            gt_map_3 = junction_map
            gt_map,_ = torch.max(junction_map[:,0:2,:,:],dim=1)
            gt_map = gt_map.unsqueeze(dim=1)
            labels_map = gt_map.to(device)

            pred_Vess, mask_pred_ori_map, mask_pred_ori_det = model(input_1,input_2,input_3)

            imgName = dataloader.dataset.getFileName()
            pred_Vess = pred_Vess.squeeze().cpu().numpy()
            labels_Vess = vessel_w.squeeze().numpy()

            image = torch.squeeze(input_1).cpu().numpy()
            image = (image.transpose((1,2,0))*255).astype('uint8')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            AUC, DC, Acc, Sen, JS = testSeg(pred_Vess, labels_Vess, imgName,image,'Vessel')

            funcMtx_Vess['AUC'].append(AUC)
            funcMtx_Vess['DC'].append(DC)
            funcMtx_Vess['Acc'].append(Acc*100)
            funcMtx_Vess['Sen'].append(Sen) 
            funcMtx_Vess['JS'].append(JS)

            #junction
            mask_pred_ori_det = mask_pred_ori_det.permute([0,2,3,1])
            mask_pred_ori_map = mask_pred_ori_map.squeeze().cpu().numpy()

            gt_map = gt_map.squeeze().numpy()
            gt_det = torch.squeeze(gt_det).numpy()
            mask_pred_ori_det = mask_pred_ori_det.squeeze().cpu().numpy()

            gt_map_3 = gt_map_3.squeeze().numpy()

            BF_gt_m, CR_gt_m,BF_gt_cir,CR_gt_cir = heatmap2point(gt_map_3*255,size=(Config3In.imgSize,Config3In.imgSize))
            _, BF_pred_m, CR_pred_m, BF_cir, CR_cir = getCMBResult(mask_pred_ori_det,mask_pred_ori_map*255,thresh=Config3In.confidence, gridSize = Config3In.gridSize,size=(Config3In.imgSize,Config3In.imgSize))
            Sen, Spe, Pre, F1  = detectionMatrix(BF_pred_m, CR_pred_m, BF_gt_m, CR_gt_m, tolerance=Config3In.tolerance,size=(Config3In.imgSize,Config3In.imgSize))
            funcMtx_cmb['SEN_det'].append(Sen)
            funcMtx_cmb['PRE_det'].append(Pre)
            funcMtx_cmb['F1_det'].append(F1)

            Sen, Spe, Pre, F1 = classifyMatrixV3(BF_pred_m, CR_pred_m, BF_gt_m, CR_gt_m, tolerance=Config3In.tolerance,size=(Config3In.imgSize,Config3In.imgSize))
            funcMtx_cmb['SEN_cla'].append(Sen)
            funcMtx_cmb['PRE_cla'].append(Pre)
            funcMtx_cmb['F1_cla'].append(F1)

            pathJ = os.path.join(save_dir,'junction')
            if not os.path.isdir(pathJ):
                os.makedirs(pathJ)

            #save junction result
            BF_gt_cir = cv2.cvtColor(np.asarray(BF_gt_cir),cv2.COLOR_RGB2BGR)
            CR_gt_cir = cv2.cvtColor(np.asarray(CR_gt_cir),cv2.COLOR_RGB2BGR)

            BF_cir = cv2.cvtColor(np.asarray(BF_cir),cv2.COLOR_RGB2BGR)
            CR_cir = cv2.cvtColor(np.asarray(CR_cir),cv2.COLOR_RGB2BGR)
            print(imgName)
            cv2.imwrite(os.path.join(pathJ,'img_gt_'+imgName),(0.6*image+0.7*BF_gt_cir+0.7*CR_gt_cir).astype(np.uint8))
            cv2.imwrite(os.path.join(pathJ,'img_pred_'+imgName),(0.6*image+0.7*BF_cir+0.7*CR_cir).astype(np.uint8))

    length = len(dataloader.dataset)

    for k in funcMtx_Vess.keys():
        if 'DC' in k  or 'JS' in k or 'Acc' in k:
            viz.plot('Vessel_'+k, np.mean(funcMtx_Vess[k]))

    for k in funcMtx_cmb.keys():
        #funcMtx_cmb[k] = funcMtx_cmb[k]/length
        if 'SPE' in k:
            continue
        viz.plot(k+'_cmb', np.mean(funcMtx_cmb[k]))

    print('Vessel - Acc, JS, DC: %04f+%04f, %04f+%04f, %04f+%04f'%(np.mean(funcMtx_Vess['Acc']), np.std(funcMtx_Vess['Acc']),\
            np.mean(funcMtx_Vess['JS']), np.std(funcMtx_Vess['JS']), np.mean(funcMtx_Vess['DC']), np.std(funcMtx_Vess['DC'])))

    #print('Vessel - Acc, JS, DC: %04f, %04f, %04f'%(funcMtx_Vess['Acc'], funcMtx_Vess['JS'], funcMtx_Vess['DC']))

    print('Detection Sen, Pre, F1: %04f+%04f, %04f+%04f, %04f+%04f'%(np.mean(funcMtx_cmb['SEN_det']), np.std(funcMtx_cmb['SEN_det']),\
            np.mean(funcMtx_cmb['PRE_det']), np.std(funcMtx_cmb['PRE_det']), np.mean(funcMtx_cmb['F1_det']), np.std(funcMtx_cmb['F1_det'])))
    print('Classification Sen, Pre, F1: %04f+%04f, %04f+%04f, %04f+%04f'%(np.mean(funcMtx_cmb['SEN_cla']), np.std(funcMtx_cmb['SEN_cla']),\
            np.mean(funcMtx_cmb['PRE_cla']), np.std(funcMtx_cmb['PRE_cla']), np.mean(funcMtx_cmb['F1_cla']), np.std(funcMtx_cmb['F1_cla'])))
    
    return funcMtx_Vess,funcMtx_cmb

def train_(model,dataloader_train,dataloader_test):
    train_batch = len(dataloader_train)
    test_batch = len(dataloader_test)

    optimizer = optim.Adam(myModel.parameters(), lr=Config3In.base_lr, weight_decay=Config3In.weight_decay)
    criterion_mse = nn.MSELoss()
    criterion_dectect = junctionLoss(Config3In.imgSize/Config3In.gridSize,Config3In.lossWeight[0],Config3In.lossWeight[1],Config3In.lossWeight[2])


    scheduler_steplr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=Config3In.num_epochs)
    schedulers = WarmupLR(scheduler_steplr, init_lr=1e-7, num_warmup=5, warmup_strategy='cos')

    best_metrix_FAZ = 0
    best_metrix_vessel = 0

    avg_cost = np.zeros([Config3In.num_epochs, 3], dtype=np.float32)
    lambda_weight = np.ones([3, Config3In.num_epochs])
    T = Config3In.temp
    
    best_metrix_FAZ = 0
    best_metrix_vessel = 0
    best_metrix_junc = 0
        
    for epoch in range(Config3In.num_epochs):
        index = epoch
        cost = np.zeros(2, dtype=np.float32)
        if Config3In.weight == 'dwa':
            if index == 0 or index == 1:
                lambda_weight[:, epoch] = 1.0
            else:
                w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
                w_2 = avg_cost[index - 1, 1] / avg_cost[index - 2, 1]
                lambda_weight[0, index] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) )
                lambda_weight[1, index] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                
        schedulers.step(epoch)
        print('Epoch %d/%d' % (epoch+1, Config3In.num_epochs))
        print('-'*10)

        print('Start detection training~')
        epoch_loss = 0
        step = 0
        dt_size = len(dataloader_train.dataset)

        for image, junction_map, junction_det, FAZ, vessels in dataloader_train:
            step += 1
            
            input_1, input_2, input_3 = image # #test

            input_1 = input_1.to(device)
            input_2 = input_2.to(device)
            input_3 = input_3.to(device)


            gt_det = junction_det
            labels_det = gt_det.to(device)
            gt_map,_ = torch.max(junction_map[:,0:2,:,:],dim=1)
            gt_map = gt_map.unsqueeze(dim=1)
            labels_map = gt_map.to(device)

            vessel_w = vessels.to(device)

            optimizer.zero_grad()
            outputs_vessel, outputs_map, outputs_det = model(input_1,input_2,input_3)

            show_image = input_1* 255.
            viz.img(name='images', img_=show_image[0, :, :, :])
            viz.img(name='label_vessel', img_=vessel_w[0, :, :, :])
            viz.img(name='prediction_vessel', img_=outputs_vessel[0, :, :, :])

            viz.img(name='labels_map', img_=labels_map[0, :, :, :])
            viz.img(name='prediction_map', img_=outputs_map[0, :, :, :])


            outputs_det = outputs_det.permute([0,2,3,1])
            gt_det = gt_det.numpy()
            mask_pred_ori_det = outputs_det.detach().cpu().numpy()
            gtShow,_,_ = showPrediction(gt_det[0,:,:],thresh=Config3In.confidence, gridSize = Config3In.gridSize,size = (Config3In.imgSize,Config3In.imgSize))
            predShow,_,_ = showPrediction(mask_pred_ori_det[0,:,:],thresh=Config3In.confidence, gridSize = Config3In.gridSize,size = (Config3In.imgSize,Config3In.imgSize))

            trans_tensor = transforms.ToTensor()
            gtShow = trans_tensor(gtShow)
            predShow = trans_tensor(predShow)

            viz.img(name='labels_det', img_=gtShow)
            viz.img(name='prediction_det', img_=predShow)

            loss_det = criterion_dectect(outputs_det, labels_det)
            loss_map = criterion_mse(outputs_map, labels_map)
            loss_junc = loss_det*0.0005 + loss_map

            l_mse = criterion_mse(outputs_vessel, vessel_w)
            l_dice = dice_coeff_loss(outputs_vessel, vessel_w)
            # loss_vessel = l_mse + l_dice*0.4
            loss_vessel = l_mse
            
            #loss = loss_faz + loss_vessel*0.3

            train_loss = [loss_vessel, loss_junc]
            loss = sum([lambda_weight[i, index] * train_loss[i] for i in range(2)])
            cost[0] = train_loss[0].item()
            cost[1] = train_loss[1].item()
            avg_cost[index, :2] += cost[:2] / train_batch

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            print("%d/%d,train_loss:%0.4f" % (step, (dt_size - 1) // dataloader_train.batch_size + 1, loss.item()))
            viz.plot('loss_vessel', loss_vessel.item())
            viz.plot('loss_junc', loss_junc.item())
            viz.plot('loss', loss.item())

        current_lr = get_lr(optimizer)
        viz.plot('learning rate', current_lr)

        if bool(epoch%3)  is False :
            save = True
            if epoch > Config3In.num_epochs-2:
                save = True

            metrix = test_(model,dataloader_test, Config3In.taskName, isSave=save)
            metrix_vessel, metrix_junc = metrix
            model.train(mode=True)

        
        if np.mean(metrix_vessel['DC']) > best_metrix_vessel and epoch >300:
            save_path = os.path.join(save_dir, 'state-{}-vess-{}.pth'.format(epoch + 1,np.mean(metrix_vessel['DC'])))
            best_metrix_vessel = np.mean(metrix_vessel['DC'])
            torch.save(model, save_path)

        if np.mean(metrix_junc['F1_det']) > best_metrix_junc and epoch >300:
            save_path = os.path.join(save_dir, 'state-{}-junc-{}.pth'.format(epoch + 1,np.mean(metrix_junc['F1_det'])))
            best_metrix_junc = np.mean(metrix_junc['F1_det'])
            torch.save(model, save_path)
            
        if epoch > Config3In.num_epochs-3:  
            save_path = os.path.join(save_dir, 'state-{}-junc-{}.pth'.format(epoch + 1,np.mean(metrix_junc['F1_det'])))  
            torch.save(model, save_path)


if __name__ =='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = Config3In.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = Config3In.savePath
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    myModel = VAFFNet4Drive()
    
    total = sum([param.nelement() for param in myModel.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))

    myModel = torch.nn.DataParallel(myModel).cuda()

    viz = Visualizer(env=Config3In.env,port=2333)

    #dataloader
    dataTrain = datasetMT_drive(root = Config3In.dataPath,sigma=Config3In.sigma,size=Config3In.gridSize, isTraining=True, imgSize = Config3In.imgSize)
    dataloader_Train = DataLoader(dataTrain, batch_size=Config3In.batch_size,shuffle=True, worker_init_fn=worker_init_fn)

    dataTest = datasetMT_drive(root = Config3In.dataPath,sigma=Config3In.sigma,size=Config3In.gridSize, isTraining=False, imgSize = Config3In.imgSize)
    dataloader_Test = DataLoader(dataTest, batch_size=1,shuffle=True, worker_init_fn=worker_init_fn)

    train_(myModel,dataloader_Train,dataloader_Test)

    


    # #test
    # path = '/home/imed/personal/kevin/result/multi-task/MT_Drive-our2/state-472-vess-79.21747276210415.pth'
    # model = torch.load(path)
    # save_dir =  Config3In.savePath


    # if isinstance(model,torch.nn.DataParallel):
    #     model = model.module

    # save_dir = Config3In.savePath
    # metrix = test_(model,dataloader_Test, Config3In.taskName, isSave=True)
    # metrix_vessel, metrix_junc = metrix
