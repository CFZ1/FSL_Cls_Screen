# 从01base复制而来，因为引用的时候不允许数字在前面
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_outRelu import resnet18
import math

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

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
        if self.args.debug3_maskLoss: #-----------------------debug3_maskLoss_s1
            kernel_size = 3
            self.conv_mask = nn.Sequential(
                nn.Conv2d(self.num_features, 1, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False),
                nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True)) #可学习参数
            
    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'euclidean' in self.mode:
            x = -euclidean_dist(x,self.fc.weight)
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
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

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
        new_fc=[]
        for class_index in class_list:
            data_index = torch.nonzero(label == class_index,as_tuple=False).squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

