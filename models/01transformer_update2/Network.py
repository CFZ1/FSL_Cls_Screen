import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_outRelu import resnet18
import math
import numpy as np

def euclidean_dist(x, y, mask=None):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    if mask==None:
        return torch.pow(x - y, 2).sum(2)
    else:
        return torch.mul(torch.pow(x - y, 2),mask.unsqueeze(0)).sum(2)

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.procedure in ['multiLabel','singleLabel1']:
            self.encoder = resnet18(True, args)
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        #---Different from 01base_episode---3
        if self.args.debug3_maskLoss:
            kernel_size = 3
            self.conv_mask = nn.Sequential(
                nn.Conv2d(self.num_features, 1, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False),
                nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True)) #可学习参数
        #---Different from 01base_episode---3
        if self.args.debug1_relationAdd and (self.args.episode_shot == 1):
            self.args.debug1_relationAdd = False
        if self.args.debug1_relationAdd:#-------------debug1_relationAdd--s1
            self.relation_lambda = nn.Linear(self.num_features, self.num_features, bias=True)
        if self.args.debug2_relationMask:
            self.protos_mask = torch.ones([self.args.num_classes,self.num_features]).detach()
            self.mask_lambda = nn.Linear(self.num_features, self.num_features, bias=True)
        
    def forward_proto(self, img_feas, label, proto_label): 
        '''
        img_feas: [bs, d]
        label: [bs] if singleLabel1
        proto_label: [num_cls] if singleLabel1
        '''
        if self.args.procedure == 'singleLabel1': #要求标签连续，即中间类别全部存在示例
            max_cls = int(max(proto_label)) + 1
            label = F.one_hot(label, num_classes=max_cls).to(device=img_feas.device).float() #因此cuda不支持long矩阵的乘积
            proto_label = F.one_hot(proto_label, num_classes=max_cls).to(device=img_feas.device).float() 
            relation1 = (proto_label @ label.transpose(-2, -1)).unsqueeze(2) #relationship between [num_cls,bs,1]
            
            bs = img_feas.shape[0]
            if self.args.debug1_relationAdd:
                if 'euclidean' in self.args.base_mode:
                    #[bs,bs,512]
                    sim_matrix = -(self.args.temperature *torch.pow(img_feas.unsqueeze(1) - img_feas.unsqueeze(0), 2))
                elif 'cos' in self.args.base_mode: 
                    #[bs,bs,512]
                    sim_matrix = torch.mul(self.args.temperature *F.normalize(img_feas, p=2, dim=-1).unsqueeze(1), F.normalize(img_feas, p=2, dim=-1).unsqueeze(0))  
                # 排除掉自己
                pos_mask = (label @ label.transpose(-2, -1)).unsqueeze(2) #[bs,bs,1]
                relation2=torch.mul(pos_mask,sim_matrix) #[bs,bs,512]
                relation2 = F.adaptive_avg_pool1d(relation2.transpose(-2, -1), 1).squeeze(-1) #[bs,512]
                #F:[bs,512]--->[bs,512] ,512通道
                relation2 = F.linear(relation2,self.relation_lambda.weight.to(device=relation2.device),self.relation_lambda.bias.to(device=relation2.device))
                relation2 = torch.sigmoid(relation2)
                #[num_cls,bs,1]*[1,bs,512]->.sum=[num_cls,bs,512]
                relation2 = relation1*(relation2.unsqueeze(0)) 
                # [num_cls,bs,512]*[num_cls,bs,512]->.sum=[num_cls,512]
                proto_list = (torch.mul(relation2,img_feas.unsqueeze(0)).sum(1))/(relation2.sum(1))
                # [num_cls,bs,1]*[num_cls,bs,512]=[num_cls,bs,512]
            else:
                #[num_cls,bs,1]*[num_cls,bs,512]->.sum=[num_cls,512]
                proto_list = (torch.mul(relation1,img_feas.unsqueeze(0)).sum(1))/(relation1.sum(1)) 
            
            if self.args.debug2_relationMask:
                num_base = max_cls-proto_label.shape[0]
                prototypes = proto_list
                if (num_base)>0:
                    prototypes = torch.cat([prototypes,self.fc.weight.data.to(device=img_feas.device)[:num_base]],dim=0) #[num_cls+c,512]
                if 'euclidean' in self.args.base_mode:
                    prototypes_sim_matrix = -(self.args.temperature *torch.pow(prototypes.unsqueeze(1) - prototypes.unsqueeze(0), 2))#[num_cls+c,num_cls+c,512],范围为(0,0.5)  
                elif 'cos' in self.args.base_mode: 
                    #[num_cls+c,num_cls+c,512]
                    prototypes_sim_matrix = torch.mul(self.args.temperature *F.normalize(prototypes, p=2, dim=-1).unsqueeze(1), F.normalize(prototypes, p=2, dim=-1).unsqueeze(0))      
                neg_mask = 1-torch.eye(prototypes.shape[0], device=prototypes_sim_matrix.device).unsqueeze(2)
                proto_list_mask_neg = torch.mul(neg_mask,prototypes_sim_matrix) #[num_cls+c,num_cls+c,512]
                # [num_cls+c,512]
                proto_list_mask_neg = F.adaptive_avg_pool1d(proto_list_mask_neg.transpose(-2, -1), 1).squeeze(-1) 
                #----------add: normalize+linear
                #[num_cls+c,512]-->[num_cls+c,512]
                proto_list_mask_neg = F.linear(proto_list_mask_neg,self.mask_lambda.weight.to(device=proto_list_mask_neg.device),self.mask_lambda.bias.to(device=proto_list_mask_neg.device))
                
                proto_list_mask = torch.ones_like(proto_list_mask_neg).detach()-proto_list_mask_neg.detach()

            else:
                proto_list_mask = None
        return proto_list, proto_list_mask
    
    def forward_logits(self, x, proto_list_mask):

        if 'cos' in self.mode:
            if self.args.debug2_relationMask:
                x = F.linear(F.normalize(x, p=2, dim=-1), proto_list_mask*F.normalize(self.fc.weight[:proto_list_mask.shape[0]], p=2, dim=-1))
            else:
                x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'euclidean' in self.mode:
            if self.args.debug2_relationMask:
                x = -euclidean_dist(x,self.fc.weight[:proto_list_mask.shape[0]],proto_list_mask)
            else:
                x = -euclidean_dist(x,self.fc.weight)
            x = self.args.temperature * x
        return x

    def forward_metric(self, x):
        x,_ = self.encode(x)
        if self.args.debug2_relationMask:
            proto_list_mask = self.protos_mask.to(device=self.fc.weight.device)
        else:
            proto_list_mask = None
        x = self.forward_logits(x,proto_list_mask)
        return x

    def encode(self, x):
        x = self.encoder(x)
        #---Different from 01base_episode---2
        if self.args.debug3_maskLoss:
            mask_sigmoid = torch.sigmoid(self.conv_mask(x)) #[bs,1,w,h]
            if self.args.mask_normalize: 
                eps = 1e-5 
                mask_sigmoid_normalize = (mask_sigmoid+eps)/((mask_sigmoid+eps).sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1))
                x = x*mask_sigmoid_normalize #[bs,512,w,h]
            else:
                x = x*mask_sigmoid
        else:
            mask_sigmoid = None
        #---Different from 01base_episode---2
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x, mask_sigmoid #---Different from 01base_episode---4

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input, mask_sigmoid = self.encode(input)
            return input, mask_sigmoid
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data)[0].detach() #---Different from 01base_episode---4

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode: 
            print('not implement')

    def update_fc_avg(self,data,label,class_list):
        proto,proto_list_mask = self.forward_proto(data,label,torch.from_numpy(class_list))
        for i,class_index in enumerate(class_list):
            self.fc.weight.data[class_index]=proto[i]
        if self.args.debug2_relationMask:
            for i in range(self.args.num_classes):
                self.protos_mask[i]=proto_list_mask[i]
        return proto
