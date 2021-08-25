#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :detectLoss.py
@Description :关键点检测loss,参考yolo
@Time        :2020/11/04 16:51:37
@Author      :Jinkui Hao
@Version     :1.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class junctionLoss(nn.Module):
    def __init__(self,S,l_coord,l_noobj,l_class):
        #@S: gridNum
        #@l_coord: weight of coordination
        super(junctionLoss,self).__init__()
        self.S = S  #girdNum
        self.l_coord = l_coord  #有目标的权重
        self.l_noobj = l_noobj  #无目标的权重
        self.l_class = l_class

    def forward(self,pred_tensor,target_tensor):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,3+3=6) [x,y,c]，[one-hot]
        target_tensor: (tensor) size(batchsize,S,S,6)
        '''
        #pred_tensor = pred_tensor.permute([0,2,3,1])
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:,:,:,2] > 0  
        noo_mask = target_tensor[:,:,:,2] == 0

        coo_mask = coo_mask.bool()
        noo_mask = noo_mask.bool()

        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        coo_pred = pred_tensor[coo_mask].view(-1,6) #提取所有有目标的区域
        point_pred = coo_pred[:,:3].contiguous().view(-1,3) #point[x1,y1,c1]
        class_pred = coo_pred[:,3:]       #预测的类别  3              
        
        coo_target = target_tensor[coo_mask].view(-1,6)
        point_target = coo_target[:,:3].contiguous().view(-1,3)
        class_target = coo_target[:,3:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,6) #没有目标的区域
        noo_target = target_tensor[noo_mask].view(-1,6)

        nooobj_loss = F.mse_loss(noo_pred[:,2],noo_target[:,2].float(),reduction='sum')

        cooobj_loss_c = F.mse_loss(point_pred[:,2],point_target[:,2].float(),reduction='sum')
        cooobj_loss_xy = F.mse_loss(point_pred[:,0:2],point_target[:,0:2].float(),reduction='sum')
        #class loss
        class_loss = F.mse_loss(class_pred,class_target.float(),reduction='sum')
       
        #no xy loss
        return (self.l_coord*cooobj_loss_c  + nooobj_loss + self.l_class*class_loss)/N

        #xy loss
        #return (self.l_coord*cooobj_loss_xy + 2*cooobj_loss_c  + self.l_noobj*nooobj_loss + class_loss)/N
