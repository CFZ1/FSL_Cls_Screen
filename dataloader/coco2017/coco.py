#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 19:36:56 2022

@author: XXX
"""

import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import random

base_cls = ["bottle","cup","knife","bowl"] 
novel_cls = ["wine_glass","fork","spoon"] 
# base_cls = ["bottle","wine_glass","cup","fork"] 
# novel_cls = ["knife","spoon","bowl"] 
Single = True #新类别的支持集使用单标签
Short="_short" #"_short" or ""

class Coco(Dataset):
    def __init__(self, root='/media/XXX/Elements/dataset/coco2017', train=True,
                 transform=None,
                 index_path=None,base_sess=None,autoaug=1,novel=False,onlyNormal=False,shot=None,DEBUG=False,start_class=None):
        
        self.base_cls = base_cls
        self.novel_cls = novel_cls
        self.way = len(novel_cls)
        self.crop_size = 224
        self.random_angle = 10.0
        self.random_scale = 0.3
        
        self.data = []
        self.targets = []
        if train:
            self.img_path = osp.join(root,"train_multiLabel")
            if base_sess: #train
                cvs_path = osp.join(root,"t_multiLabel{}.csv".format(Short))
                self.data, self.targets = self.readData(cvs_path, base_sess) # base_cls
                if DEBUG: #--------------------------------
                    print('----------------we are in debug mode----------------')
                    self.data = random.sample(self.data, 20)
                    self.targets = random.sample(self.targets, 20)
                
            else: #3-way, 5-shot
                cvs_path = osp.join(root,"t_multiLabel{}.csv".format(Short))
                data, targets = self.readData(cvs_path, base_sess)
                if shot==None:
                    print('Please specify parameter **shot**')
                self.target_use = [torch.tensor(x) for x in targets]   
                label_list = torch.stack(self.target_use, dim=0)
                self.data = []
                self.targets = []
                if start_class is None:
                    start_class = len(base_cls)
                for class_index in range(start_class,len(base_cls+novel_cls)): #way*shot
                    if Single:
                        data_index = torch.nonzero((label_list[:,class_index] == 1) &(label_list.sum(-1)==1),as_tuple=False)[:shot]
                    else:
                        data_index = torch.nonzero(label_list[:,class_index] == 1,as_tuple=False)[:shot]
                    self.data.extend([data[inds] for inds in data_index.squeeze(-1)])
                    self.targets.extend([targets[inds] for inds in data_index.squeeze(-1)])
        else:
            self.img_path = osp.join(root,"val_multiLabel")
            if base_sess:
                cvs_path = osp.join(root,"v_multiLabel.csv")
                self.data, self.targets = self.readData(cvs_path, base_sess) # base_cls
                if DEBUG:
                    self.data = random.sample(self.data, 4)
                    self.targets = random.sample(self.targets, 4)
                
            else:
                cvs_path = osp.join(root,"v_multiLabel.csv")
                self.data, self.targets = self.readData(cvs_path, base_sess) #novel_cls + base_cls
                if DEBUG:
                    self.data = random.sample(self.data, 40)
                    self.targets = random.sample(self.targets, 40)
        self.target_use = [torch.tensor(x) for x in self.targets]       
        if autoaug==0:
            
            self.transform = transforms.Compose([
                    transforms.Resize(self.crop_size),
                    transforms.CenterCrop(self.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
                if train:
                    self.transform = transforms.Compose([
                        transforms.Resize(self.crop_size),
                        transforms.RandomRotation(degrees=self.random_angle, resample=Image.BILINEAR),
                        transforms.RandomResizedCrop(
                            size=self.crop_size, scale=(1-self.random_scale, 1+self.random_scale), ratio=(1.0, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(self.crop_size),
                        transforms.CenterCrop(self.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])          
                               
                
    def rand_sort(self,data,targets):
        """
        同时将两个list按照相同的顺序打乱
        """
        cc = list(zip(data, targets))
        random.shuffle(cc)
        data[:], targets[:] = zip(*cc)
        return data,targets                
    def readData(self,cvs_path, base_sess):
        
        train_data = pd.read_csv(cvs_path)
        # #novel classes移动到最后几列
        novel_classes = self.novel_cls
        base_classes = self.base_cls
        for filter_cls in base_classes+novel_classes:
            bubble = train_data.pop(filter_cls)
            train_data.insert(loc=train_data.shape[1], column=filter_cls, value=bubble, allow_duplicates=False)
        
        train_data = np.array(train_data).tolist()
            
        if base_sess:
            for class_index in range(len(base_classes),len(base_classes+novel_classes)):
                train_data = [data for data in train_data if data[class_index+1] == 0]
            data = [osp.join(self.img_path,data_row[0]) for data_row in train_data]
            targets = [ data_row[1:-len(novel_classes)] for data_row in train_data] #base_cls
        else:
            data = [osp.join(self.img_path,data_row[0]) for data_row in train_data]
            targets = [ data_row[1:] for data_row in train_data] #base_cls+novel_cls
        return data,targets
        
        
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):

        path, targets = self.data[i], self.target_use[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets
    
    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            np.ndarray: Image categories of specified index.
        """
        gt_labels = self.target_use[idx]
        cat_ids = torch.where(gt_labels == 1)[0]
        cat_ids = cat_ids.numpy()
        return cat_ids
    
def get_base_dataloader(args):
    
    trainset = Coco(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug)
    testset = Coco(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug)
    
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    
    trainset = Coco(root=args.dataroot, train=True, base_sess=False,shot = args.shot,DEBUG=args.debug)
    
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)
        
    testset = Coco(root=args.dataroot, train=False,base_sess=False,DEBUG=args.debug) #base+novel
    
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    return trainset, trainloader, testloader

class CategoriesSampler():

    def __init__(self, args,label, n_batch, n_cls, n_per, ):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class,[normal,'pinhole','scratch','tin']
        if args.use_back: #
            for i in range(len(label[0]) + 1):
                if i==0:
                    ind = np.argwhere(label.sum(1) == 0).reshape(-1)
                else:
                    ind = np.argwhere(label[:,i-1] == 1).reshape(-1)  # all data index of this class
                self.m_ind.append(ind.tolist())
        else:
            for i in range(len(label[0])):
                ind = np.argwhere(label[:,i] == 1).reshape(-1)  # all data index of this class
                self.m_ind.append(ind.tolist())

    def __len__(self):
        return self.n_batch

    def __iter__(self):

        for i_batch in range(self.n_batch):
            batch = []
            if len(self.m_ind) < self.n_cls:
                print('error!!! we only have {} classes, but you want to sample {} classes'
                      .format(len(self.m_ind),self.n_cls))
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # sample n_cls classes from total classes.
            for c in classes:
                l = self.m_ind[c]  # all data indexs of this class
                if batch != []: #因为是多标签，过滤掉已经选择的样本
                    temp = set(np.array(batch).flatten().tolist())
                    l = list(set(l)-temp)
                pos = random.sample(l,self.n_per)
                batch.append(pos)
            batch = torch.from_numpy(np.array(batch)).t().reshape(-1)
            yield batch
            
def get_base_dataloader_meta(args):
    trainset = Coco(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug)
    testset = Coco(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug)

    sampler = CategoriesSampler(args,trainset.targets, args.train_episode, args.episode_way,
                                    args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader_meta(args):
    trainset = Coco(root=args.dataroot, train=True, base_sess=False,shot = args.episode_shot,DEBUG=args.debug,start_class = 0)
    
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        supportLoader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        supportLoader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)
        
    testset = Coco(root=args.dataroot, train=False,base_sess=False,DEBUG=args.debug) #base+novel
    queryLoader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    
    return trainset, supportLoader, queryLoader


if __name__ == '__main__':
    import argparse
    dataroot = '/media/XXX/new/dataset/mobile_screen/0multi_label'
    # trainset = mobileScreen(root=dataroot, train=True,base_sess=True)
    # # cls = np.unique(trainset.targets)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=2, shuffle=True, num_workers=0,
    #                                           pin_memory=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataroot', type=str, default="/media/XXX/Elements/dataset/coco2017")
    parser.add_argument('-use_back',action='store_true', default=False)

    parser.add_argument('-train_episode', type=int, default=100)
    parser.add_argument('-episode_shot', type=int, default=2)
    parser.add_argument('-episode_way', type=int, default=2)
    parser.add_argument('-episode_query', type=int, default=3)
    parser.add_argument('-test_batch_size', type=int, default=2)
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-debug',action='store_true', default=False)
    args = parser.parse_args()
    trainset, trainloader, testloader = get_base_dataloader_meta(args)

    for i, batch in enumerate(trainloader, 1):
        print("hello")