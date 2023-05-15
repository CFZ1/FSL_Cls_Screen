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

