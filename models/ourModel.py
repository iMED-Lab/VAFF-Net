import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torchvision import models
from functools import partial


nonlinearity = partial(F.relu, inplace=True)
        
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch,layer_num,isMaxpooling = False):
        super(ConvBlock, self).__init__()
        self.ismax = isMaxpooling
        layers = []
        for i in range(layer_num):
            conv2d = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.conv_seque = nn.Sequential(*layers)
        self.maxPooling = nn.MaxPool2d(kernel_size=2, stride=2) 
        
    def  forward(self, x):
        x = self.conv_seque(x)
        
        if self.ismax:
            return self.maxPooling(x)
        else:
            return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def  forward(self, x):

        return self.conv(x)
       
#our model
class VAFFNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(VAFFNet, self).__init__()

        resnet = models.resnet50(pretrained=True)
        
        #self.firstconv = resnet.conv1
        self.firstconv_1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.firstbn_1 = resnet.bn1
        self.firstrelu_1 = resnet.relu
        self.firstdownsample_1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encoder1_1 = resnet.layer1
        

        #2
        self.firstconv_2 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.firstbn_2 = resnet.bn1
        self.firstrelu_2 = resnet.relu
        self.firstdownsample_2 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encoder1_2 = resnet.layer1

        #3
        self.firstconv_3 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.firstbn_3 = resnet.bn1
        self.firstrelu_3 = resnet.relu
        self.firstdownsample_3 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encoder1_3 = resnet.layer1

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        #heatmap branch
        # 1*1 layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

        self.cat_conv = nn.Conv2d(256*5, 256, 3, 1, 1)

        self.gate_conv_FAZ = nn.Sequential(
            nn.Conv2d(64*3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.gate_conv_vessel = nn.Sequential(
            nn.Conv2d(64*3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.gate_conv_juncM = nn.Sequential(
            nn.Conv2d(64*3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        #FAZ
        self.finaldeconv1 = nn.Conv2d(256, 256, 3, 1, 1)  #152*152
        self.finalBN = nn.BatchNorm2d(256)
        self.finalrelu = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(256, out_ch, kernel_size=1, stride=1, padding=0)

        #vessel
        self.finaldeconv1_v = nn.Conv2d(256, 256, 3, 1, 1) 
        self.finalBN_v = nn.BatchNorm2d(256)
        self.finalrelu_v = nn.ReLU(inplace=True)
        self.finalconv2_v = nn.Conv2d(256, out_ch, kernel_size=1, stride=1, padding=0)

        #heatmap branch
        self.finaldeconv1_map = nn.Conv2d(256, 64, 3, 1, 1)  #152*152
        self.finalBN_map = nn.BatchNorm2d(64)
        self.finalrelu_map = nn.ReLU(inplace=True)
        self.finalconv2_map = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        #detetion
         #self.finaldeconv1 = nn.Conv2d(256, 256, 3, 1, 1)
        #self.finaldeconv1_d = nn.Conv2d(256, 128, 4, 2, 1)  #grid=4\
        self.finaldeconv1_d = nn.Conv2d(256, 128, 6, 4, 1)  #grid=8
        self.finalBN_d = nn.BatchNorm2d(128)
        self.finalrelu_d = nn.ReLU(inplace=True)

        self.finaldeconv1_d2 = nn.Conv2d(128, 128, 4, 2, 1)  
        self.finalBN_d2 = nn.BatchNorm2d(128)
        self.finalrelu_d2 = nn.ReLU(inplace=True)
        
        self.finalconv2_d = nn.Conv2d(128, 6, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def _upsample_(self, x, y):
        _,_,H,W = y.size()
        out = F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True)

        return out

    def forward(self, x_w,x_d,x_s):
        # x_w
        x_w = self.firstconv_1(x_w)
        x_w = self.firstbn_1(x_w)
        x_w = self.firstrelu_1(x_w)
        down_x_w = self.firstdownsample_1(x_w)
        # e1_w = self.encoder1_1(down_x_w)
        e1_w = self.encoder1(down_x_w)
        e2_w = self.encoder2(e1_w)
        e3_w = self.encoder3(e2_w)
        e4_w = self.encoder4(e3_w)
        p5_w = self.toplayer(e4_w)
        p4_w = self._upsample_add(p5_w, self.latlayer1(e3_w))
        p3_w = self._upsample_add(p4_w, self.latlayer2(e2_w))
        p2_w = self._upsample_add(p3_w, self.latlayer3(e1_w))
        p1_w = self._upsample_add(p2_w, self.latlayer4(x_w))

        # x_d
        x_d = self.firstconv_2(x_d)
        x_d = self.firstbn_2(x_d)
        x_d = self.firstrelu_2(x_d)
        down_x_d = self.firstdownsample_2(x_d)
        # e1_d = self.encoder1_2(down_x_d)
        e1_d = self.encoder1(down_x_d)
        e2_d = self.encoder2(e1_d)
        e3_d = self.encoder3(e2_d)
        e4_d = self.encoder4(e3_d)
        p5_d = self.toplayer(e4_d)
        p4_d = self._upsample_add(p5_d, self.latlayer1(e3_d))
        p3_d = self._upsample_add(p4_d, self.latlayer2(e2_d))
        p2_d = self._upsample_add(p3_d, self.latlayer3(e1_d))
        p1_d = self._upsample_add(p2_d, self.latlayer4(x_d))

        # x_s
        x_s = self.firstconv_3(x_s)
        x_s = self.firstbn_3(x_s)
        x_s = self.firstrelu_3(x_s)
        down_x_s = self.firstdownsample_3(x_s)
        # e1_s = self.encoder1_3(down_x_s)
        e1_s = self.encoder1(down_x_s)
        e2_s = self.encoder2(e1_s)
        e3_s = self.encoder3(e2_s)
        e4_s = self.encoder4(e3_s)
        p5_s = self.toplayer(e4_s)
        p4_s = self._upsample_add(p5_s, self.latlayer1(e3_s))
        p3_s = self._upsample_add(p4_s, self.latlayer2(e2_s))
        p2_s = self._upsample_add(p3_s, self.latlayer3(e1_s))
        p1_s = self._upsample_add(p2_s, self.latlayer4(x_s))

        gate_FAZ = self.gate_conv_FAZ(torch.cat([x_w,x_d,x_s],dim=1))
        gate_vessel = self.gate_conv_vessel(torch.cat([x_w,x_d,x_s],dim=1))
        gate_juncM = self.gate_conv_juncM(torch.cat([x_w,x_d,x_s],dim=1))
        #gate_juncR = self.gate_conv_juncR(torch.cat([x_w,x_d,x_s],dim=1))

        p_FAZ = gate_FAZ[:,0,:,:].unsqueeze(1)*p1_w + gate_FAZ[:,1,:,:].unsqueeze(1)*p1_d + gate_FAZ[:,2,:,:].unsqueeze(1)*p1_s
        p_vessel = gate_vessel[:,0,:,:].unsqueeze(1)*p1_w + gate_vessel[:,1,:,:].unsqueeze(1)*p1_d + gate_vessel[:,2,:,:].unsqueeze(1)*p1_s
        p_juncM = gate_juncM[:,0,:,:].unsqueeze(1)*p1_w + gate_juncM[:,1,:,:].unsqueeze(1)*p1_d + gate_juncM[:,2,:,:].unsqueeze(1)*p1_s
        #p_juncR = gate_juncR[:,0,:,:].unsqueeze(1)*p2_w + gate_juncR[:,1,:,:].unsqueeze(1)*p2_d + gate_juncR[:,2,:,:].unsqueeze(1)*p2_s

        # p = torch.cat([p4,p3,p2,p1,px],dim=1)
        # p = self.cat_conv(p)

        #faz
        out_faz = self.finaldeconv1(p_FAZ)
        out_faz = self.finalBN(out_faz)
        out_faz = self.finalrelu(out_faz)
        out_faz = self.finalconv2(out_faz)
        out_faz = nn.Sigmoid()(out_faz)

        #vessel
        out_v = self.finaldeconv1_v(p_vessel)
        out_v = self.finalBN_v(out_v)
        out_v = self.finalrelu_v(out_v)
        out_v = self.finalconv2_v(out_v)
        out_v = nn.Sigmoid()(out_v)

        #junction map
        out_map = self.finaldeconv1_map(p_juncM)
        out_map = self.finalBN_map(out_map)
        out_map = self.finalrelu_map(out_map)
        out_map = self.finalconv2_map(out_map)
        out_map = nn.Sigmoid()(out_map)

        #detection
        out_d = self.finaldeconv1_d(p_juncM)
        out_d = self.finalBN_d(out_d)
        out_d = self.finalrelu_d(out_d)
        out_d = self.finaldeconv1_d2(out_d)
        out_d = self.finalBN_d2(out_d)
        out_d = self.finalrelu_d2(out_d)
        out_d = self.finalconv2_d(out_d)
        
        return  out_faz,out_v,out_map, out_d


class VAFFNet4Drive(nn.Module):
    #用于heatmap预测
    def __init__(self, in_ch=3, out_ch=1):
        super(VAFFNet4Drive, self).__init__()

        resnet = models.resnet50(pretrained=True)
        
        #self.firstconv = resnet.conv1
        self.firstconv_1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.firstbn_1 = resnet.bn1
        self.firstrelu_1 = resnet.relu
        self.firstdownsample_1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.xavier_normal_(self.firstconv_1.weight.data)
        nn.init.xavier_normal_(self.firstdownsample_1.weight.data)
        # self.encoder1_1 = resnet.layer1
        

        #2
        self.firstconv_2 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.firstbn_2 = resnet.bn1
        self.firstrelu_2 = resnet.relu
        self.firstdownsample_2 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.normal_(self.firstconv_2.weight.data)
        nn.init.normal_(self.firstdownsample_2.weight.data)
        # self.encoder1_2 = resnet.layer1

        #3
        self.firstconv_3 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.firstbn_3 = resnet.bn1
        self.firstrelu_3 = resnet.relu
        self.firstdownsample_3 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encoder1_3 = resnet.layer1

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        #heatmap branch
        # 1*1 layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

        self.cat_conv = nn.Conv2d(256*5, 256, 3, 1, 1)

        self.gate_conv_vessel = nn.Sequential(
            nn.Conv2d(64*3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        self.gate_conv_juncM = nn.Sequential(
            nn.Conv2d(64*3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        #vessel
        self.finaldeconv1_v = nn.Conv2d(256, 256, 3, 1, 1) 
        self.finalBN_v = nn.BatchNorm2d(256)
        self.finalrelu_v = nn.ReLU(inplace=True)
        self.finalconv2_v = nn.Conv2d(256, out_ch, kernel_size=1, stride=1, padding=0)

        #heatmap branch
        self.finaldeconv1_map = nn.Conv2d(256, 64, 3, 1, 1)  #152*152
        self.finalBN_map = nn.BatchNorm2d(64)
        self.finalrelu_map = nn.ReLU(inplace=True)
        self.finalconv2_map = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

        #detetion
         #self.finaldeconv1 = nn.Conv2d(256, 256, 3, 1, 1)
        #self.finaldeconv1_d = nn.Conv2d(256, 128, 4, 2, 1)  #grid=4\
        self.finaldeconv1_d = nn.Conv2d(256, 128, 6, 4, 1)  #grid=8
        self.finalBN_d = nn.BatchNorm2d(128)
        self.finalrelu_d = nn.ReLU(inplace=True)

        self.finaldeconv1_d2 = nn.Conv2d(128, 128, 4, 2, 1)  
        self.finalBN_d2 = nn.BatchNorm2d(128)
        self.finalrelu_d2 = nn.ReLU(inplace=True)
        
        self.finalconv2_d = nn.Conv2d(128, 6, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def _upsample_(self, x, y):
        _,_,H,W = y.size()
        out = F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True)

        return out

    def forward(self, x_w,x_d,x_s):
        # x_w
        x_w = self.firstconv_1(x_w)
        x_w = self.firstbn_1(x_w)
        x_w = self.firstrelu_1(x_w)
        down_x_w = self.firstdownsample_1(x_w)
        # e1_w = self.encoder1_1(down_x_w)
        e1_w = self.encoder1(down_x_w)
        e2_w = self.encoder2(e1_w)
        e3_w = self.encoder3(e2_w)
        e4_w = self.encoder4(e3_w)
        p5_w = self.toplayer(e4_w)
        p4_w = self._upsample_add(p5_w, self.latlayer1(e3_w))
        p3_w = self._upsample_add(p4_w, self.latlayer2(e2_w))
        p2_w = self._upsample_add(p3_w, self.latlayer3(e1_w))
        p1_w = self._upsample_add(p2_w, self.latlayer4(x_w))

        # x_d
        x_d = self.firstconv_2(x_d)
        x_d = self.firstbn_2(x_d)
        x_d = self.firstrelu_2(x_d)
        down_x_d = self.firstdownsample_2(x_d)
        # e1_d = self.encoder1_2(down_x_d)
        e1_d = self.encoder1(down_x_d)
        e2_d = self.encoder2(e1_d)
        e3_d = self.encoder3(e2_d)
        e4_d = self.encoder4(e3_d)
        p5_d = self.toplayer(e4_d)
        p4_d = self._upsample_add(p5_d, self.latlayer1(e3_d))
        p3_d = self._upsample_add(p4_d, self.latlayer2(e2_d))
        p2_d = self._upsample_add(p3_d, self.latlayer3(e1_d))
        p1_d = self._upsample_add(p2_d, self.latlayer4(x_d))

        # x_s
        x_s = self.firstconv_3(x_s)
        x_s = self.firstbn_3(x_s)
        x_s = self.firstrelu_3(x_s)
        down_x_s = self.firstdownsample_3(x_s)
        # e1_s = self.encoder1_3(down_x_s)
        e1_s = self.encoder1(down_x_s)
        e2_s = self.encoder2(e1_s)
        e3_s = self.encoder3(e2_s)
        e4_s = self.encoder4(e3_s)
        p5_s = self.toplayer(e4_s)
        p4_s = self._upsample_add(p5_s, self.latlayer1(e3_s))
        p3_s = self._upsample_add(p4_s, self.latlayer2(e2_s))
        p2_s = self._upsample_add(p3_s, self.latlayer3(e1_s))
        p1_s = self._upsample_add(p2_s, self.latlayer4(x_s))

        gate_vessel = self.gate_conv_vessel(torch.cat([x_w,x_d,x_s],dim=1))
        gate_juncM = self.gate_conv_juncM(torch.cat([x_w,x_d,x_s],dim=1))
        #gate_juncR = self.gate_conv_juncR(torch.cat([x_w,x_d,x_s],dim=1))

        p_vessel = gate_vessel[:,0,:,:].unsqueeze(1)*p1_w + gate_vessel[:,1,:,:].unsqueeze(1)*p1_d + gate_vessel[:,2,:,:].unsqueeze(1)*p1_s
        p_juncM = gate_juncM[:,0,:,:].unsqueeze(1)*p1_w + gate_juncM[:,1,:,:].unsqueeze(1)*p1_d + gate_juncM[:,2,:,:].unsqueeze(1)*p1_s
        #p_juncR = gate_juncR[:,0,:,:].unsqueeze(1)*p2_w + gate_juncR[:,1,:,:].unsqueeze(1)*p2_d + gate_juncR[:,2,:,:].unsqueeze(1)*p2_s

        # p = torch.cat([p4,p3,p2,p1,px],dim=1)
        # p = self.cat_conv(p)

        #vessel
        out_v = self.finaldeconv1_v(p_vessel)
        out_v = self.finalBN_v(out_v)
        out_v = self.finalrelu_v(out_v)
        out_v = self.finalconv2_v(out_v)
        out_v = nn.Sigmoid()(out_v)

        #junction map
        out_map = self.finaldeconv1_map(p_juncM)
        out_map = self.finalBN_map(out_map)
        out_map = self.finalrelu_map(out_map)
        out_map = self.finalconv2_map(out_map)
        out_map = nn.Sigmoid()(out_map)

        #detection
        out_d = self.finaldeconv1_d(p_juncM)
        out_d = self.finalBN_d(out_d)
        out_d = self.finalrelu_d(out_d)
        out_d = self.finaldeconv1_d2(out_d)
        out_d = self.finalBN_d2(out_d)
        out_d = self.finalrelu_d2(out_d)
        out_d = self.finalconv2_d(out_d)
        
        return  out_v,out_map, out_d