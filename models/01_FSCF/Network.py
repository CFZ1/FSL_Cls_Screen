# import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_outRelu import resnet18
# from models.resnet20_cifar import *
# import torchvision.models as models
# from torchvision import transforms

class Mlp(nn.Module): #参考https://github.com/google-research/simclr/model_util.py:154-166
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.proj = nn.Sequential(nn.Linear(in_features, hidden_features, bias=False),
                                  nn.BatchNorm1d(hidden_features),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(hidden_features, out_features, bias=False))

    def forward(self, x):
        x = self.proj(x)
        return x

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.procedure in ['multiLabel','singleLabel1']:
            # model = models.resnet50(pretrained=True)  
            # self.encoder = nn.Sequential(*list(model.children())[:-2]) 
            # self.num_features = 2048
            # self.hidden_features = 512 
            # self.out_features = 128
            self.encoder = resnet18(True, args)
            self.num_features = 512
            self.hidden_features = 512 
            self.out_features = 128
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.g =  Mlp(in_features=self.num_features,
                      hidden_features=self.hidden_features,
                      out_features = self.out_features)
        if self.args.debug3_maskLoss: #-----------------------debug3_maskLoss_s1
            kernel_size = 3
            self.conv_mask = nn.Sequential(
                nn.Conv2d(self.num_features, 1, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False),
                nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True)) #可学习参数
    def forward_metric(self, x):
        feature,mask_sigmoid = self.encode(x)#-----------------------debug3_maskLoss_4
        out = self.g(feature)
        
        logits = F.linear(F.normalize(feature, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
        #---------------------------------temperature
        logits = self.args.temperature * logits

        return F.normalize(out, dim=-1),logits,mask_sigmoid#-----------------------debug3_maskLoss_5

    def encode(self, x):
        x = self.encoder(x)
        if self.args.debug3_maskLoss: #-----------------------debug3_maskLoss_2
            mask_sigmoid = torch.sigmoid(self.conv_mask(x)) #[bs,1,w,h]
            if self.args.mask_normalize: 
                eps = 1e-5 
                mask_sigmoid_normalize = (mask_sigmoid+eps)/((mask_sigmoid+eps).sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1))
                x = x*mask_sigmoid_normalize #[bs,512,w,h]
            else:
                x = x*mask_sigmoid
        else:
            mask_sigmoid = None
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, mask_sigmoid #-----------------------debug3_maskLoss_3

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input,_ = self.encode(input)#-----------------------debug3_maskLoss_13
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data)[0].detach()#-----------------------debug3_maskLoss_14e
            
        new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            print('not implement')

    def update_fc_avg(self,data,label,class_list):
        data = F.normalize(data, p=2, dim=-1)#---------------------------------0725--0
        new_fc=[]
        for class_index in class_list:
            data_index = torch.nonzero(label == class_index,as_tuple=False).squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

