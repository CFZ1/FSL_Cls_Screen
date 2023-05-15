import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_outRelu import resnet18
from models.resnet20_cifar import *


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.procedure in ['multiLabel','singleLabel1']:
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        
        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        
        nn.init.orthogonal_(self.fc.weight)
        self.dummy_orthogonal_classifier=nn.Linear(self.num_features, self.pre_allocate-self.args.base_class, bias=False)
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        
        self.dummy_orthogonal_classifier.weight.data=self.fc.weight.data[self.args.base_class:,:]
        print(self.dummy_orthogonal_classifier.weight.data.size())
        
        print('self.dummy_orthogonal_classifier.weight initialized over.')
        if self.args.debug3_maskLoss: #-----------------------debug3_maskLoss_s1
            kernel_size = 3
            self.conv_mask = nn.Sequential(
                nn.Conv2d(self.num_features, 1, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False),
                nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True)) #可学习参数

    def forward_metric(self, x):
        x,mask_sigmoid = self.encode(x)
        if 'cos' in self.mode:
            
            x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x2 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))
            
            x = torch.cat([x1[:,:self.args.base_class],x2],dim=1)
            
            x = self.args.temperature * x
            
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x,mask_sigmoid

    def forpass_fc(self,x):
        x,_ = self.encode(x)
        if 'cos' in self.mode:
            
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x
            
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

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
        return x, mask_sigmoid
    
    def pre_encode(self,x):
        
        if self.args.dataset in ['cifar100','manyshotcifar']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            
        elif (self.args.dataset in ['mini_imagenet','manyshotmini','cub200']) or (self.args.procedure in ['multiLabel','singleLabel1']):
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
        
        return x
        
    
    def post_encode(self,x):
        if self.args.dataset in ['cifar100','manyshotcifar']:
            
            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif (self.args.dataset in ['mini_imagenet','manyshotmini','cub200']) or (self.args.procedure in ['multiLabel','singleLabel1']):
            
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
        
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
            
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input,mask_sigmoid = self.forward_metric(input)
            return input,mask_sigmoid
        elif self.mode == 'encoder':
            input,_ = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data)[0].detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index = torch.nonzero(label == class_index,as_tuple=False).squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

