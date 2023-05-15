#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:29:21 2022

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
from dataloader.sampler import CategoriesSampler
import copy
'''
1.data_util中
1.base 中cons_mobilescreen_openness替换cons_mobilescreen
'''
#增加normal类别作为base_cls
openness = []
novel_cls_remove = ['floater', 'watermark']
base_cls = ['normal','pinhole', 'tin', 'scratch', 'bubble']
#label = 0,1,2,3,4....
novel_cls = ['floater', 'watermark', 'fragment']

base_cls = [i for i in base_cls if i not in openness]
novel_cls = openness+novel_cls
novel_cls = [i for i in novel_cls if i not in novel_cls_remove]
#label = num_base+0,num_base+1,num_base+2,.... 
if 'normal' in base_cls:
    filterNor=False 
class mobileScreenSingle(Dataset):
    def __init__(self, root='/media/XXX/Elements/dataset/mobile_screen/0_few_shot', train=True,
                 transform=None,
                 index_path=None,base_sess=None,autoaug=1,novel=False,onlyNormal=False,shot=None,DEBUG=False,test_mode='test',args=None):
        
        self.base_cls = base_cls
        self.novel_cls = novel_cls
        
        self.img_path = osp.join(root,'slice_img_v2_256')
        self.data = []
        self.targets = []
        self.origialLegth = None
        self.train = train
#         self.add_noise_ONoff = False
        if train:
            if base_sess: #train
                cvs_path = osp.join(root,"slice_img_v2_256/base_train.csv")
                self.data, self.targets = self.readData(cvs_path, base_sess,onlyNormal,filterNormal=filterNor) # base_cls
                if DEBUG: #--------------------------------
                    print('----------------we are in debug mode----------------')
                    self.data = random.sample(self.data, 100)
                    self.targets = random.sample(self.targets, 100)         
            else:
                cvs_path = osp.join(root,"slice_img_v2_256/base_train.csv")
                self.data, self.targets = self.readData(cvs_path, base_sess,onlyNormal,filterNormal=filterNor) # base_cls
                if DEBUG: #--------------------------------
                    print('----------------we are in debug mode----------------')
                    self.data = random.sample(self.data, 100)
                    self.targets = random.sample(self.targets, 100)    
                
        else:
            if base_sess:
                if test_mode == 'val':
                    cvs_path = osp.join(root,"slice_img_v2_256/base_val.csv")
                elif test_mode == 'test':
                    cvs_path = osp.join(root,"slice_img_v2_256/base_test.csv")
                self.data, self.targets = self.readData(cvs_path, base_sess,filterNormal=filterNor) # base_cls
                if DEBUG:
                    self.data = random.sample(self.data, 4)
                    self.targets = random.sample(self.targets, 4)           
            else:
                cvs_path = [osp.join(root,dir_s) for dir_s in ["slice_img_v2_256/novel.csv","slice_img_v2_256/base_test.csv"]]
                self.data, self.targets = self.readData(cvs_path, base_sess,filterNormal=filterNor) #base_cls+novel_cls  
                # if DEBUG:
                #     self.data = random.sample(self.data, 8)
                #     self.targets = random.sample(self.targets, 8) 
        if (not filterNor):
            self.target_use = [torch.tensor([1-sum(x)]+x) for x in self.targets] 
            self.targets = [int(np.where(np.array([1-sum(xx)]+xx) == 1)[0]) for xx in self.targets]
        else:
            self.target_use = [torch.tensor(x) for x in self.targets] 
            self.targets = [int(np.where(np.array(xx) == 1)[0]) for xx in self.targets]
        if autoaug==0:    
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            if DEBUG:
                if train:
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.Resize([200, 200]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize([200, 200]),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            else:
                if train:
                    if (args is not None) and (args.data_augment in ['01cec']):
                        print('data_augment in 01cec')
                        self.transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            # transforms.RandomVerticalFlip(),
                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), #----多增加一个数据增强
                            transforms.Resize([256, 256]),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                    else:                      
                        self.transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), #----多增加一个数据增强
                            transforms.Resize([256, 256]),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
                else:
#                     self.add_noise = transforms.Compose([
#                         transforms.ToTensor(),
#                         transforms.Resize([256, 256]),
#                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                     ])   
                    self.transform = transforms.Compose([
                        transforms.Resize([256, 256]),
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
             
    def csv2list(self,cvs_path):
        train_data = pd.read_csv(cvs_path)
        filter_clos = list(set(train_data.columns.tolist()[1:])-set(base_cls+novel_cls)) ###1031
        train_data = train_data.drop(columns=filter_clos) ###1031
        # novel_cls移动到最后几列
        for clsa in novel_cls:
            clsa_data  = train_data.pop(clsa)
            train_data.insert(loc=train_data.shape[1], column=clsa, value=clsa_data, allow_duplicates=False)
        train_data = np.array(train_data).tolist()
        for filter_clo in filter_clos: ###1031
            train_data = [data for data in train_data if filter_clo not in data[0]]###1031
        return train_data 
    
    def readData(self,cvs_path, base_sess,onlyNormal=False,onlySingle=True,filterNormal=True,shot=None):
        """
        onlySingle: True的情况下过滤掉所有multi-label，保留single-label，否则数据集同时存在multi-label和single-label
        filterNormal: True的情况下过滤掉所有无缺陷（正常）图片
        """
        if isinstance(cvs_path, str):
            train_data = self.csv2list(cvs_path)
        elif isinstance(cvs_path, list):
            train_data = []
            for cvs_path_ in cvs_path:
                train_data.extend(self.csv2list(cvs_path_))

        if onlyNormal:
            train_data = [data for data in train_data if 'normal' in data[0]]
        if onlySingle:
            train_data = [data for data in train_data if sum(data[1:]) < 2]
        if filterNormal:
            train_data = [data for data in train_data if 'normal' not in data[0]]   
        if openness !=[]:
            if base_sess:
                for openness_base in openness:
                    train_data = [data for data in train_data if openness_base not in data[0]]    
            elif self.train:
                train_data_final = []
                for openness_base in openness:
                    train_data_final.extend([data for data in train_data if openness_base in data[0]]) 
                train_data = train_data_final
      
        data = [osp.join(self.img_path,data_row[0]) for data_row in train_data] 
        if base_sess:
            targets = [data_row[1:len(set(base_cls)-{'normal'})+1] for data_row in train_data] #base_cls    
        else:
            targets = [data_row[1:] for data_row in train_data] #base_cls+novel_cls
        return data,targets
        
        
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
#         if self.add_noise_ONoff:
# #             wo = GaussNoiseSNR(var_limit=(1.0**2,1.0**2),p=1.0,per_channel=False)
#             wo = albumentations.GaussNoise(var_limit=(1.5**2,1.5**2),p=1.0,per_channel=False)
#             image = np.array(Image.open(path).convert('RGB'))
#             image = wo(image=image)
#             image = self.add_noise(image['image'])
#         else:
#             image = self.transform(Image.open(path).convert('RGB'))
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets
    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            np.ndarray: Image categories of specified index.
        """
        cat_ids = self.targets[idx]
        return [cat_ids]
    def get_novel_ids(self):
        idx_lists = []
        openness = [i for i in range(len(self.base_cls),5)]
        for novel_id in range(len(base_cls),len(base_cls)+len(novel_cls)):
            idx_list = []
            for idx,target_id in enumerate(self.targets):
                if target_id == novel_id:
                    if novel_id in openness:
                        if idx>self.origialLegth:
                            idx_list.append(idx)
                    else:
                        idx_list.append(idx)
            idx_lists.append(idx_list)
            
        return idx_lists                    
              
def get_base_dataloader_meta(args,session):
    trainset = mobileScreenSingle(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug,args=args)
    valset = mobileScreenSingle(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    testset = mobileScreenSingle(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='test')

    # 此处的args.episode_way = len(base_cls)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                    args.episode_shot + args.episode_query) #label = 0,1,2...0,1,2

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    valloader = torch.utils.data.DataLoader(
        dataset=valset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, valloader, testloader

# def get_new_dataloader_manyRuns(args,session):
        
#     testset = mobileScreenSingle(root=args.dataroot, train=False,base_sess=False,DEBUG=args.debug) #base+novel
#     return testset

def get_base_dataloader(args,session):
    trainset = mobileScreenSingle(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug,args=args)
    valset = mobileScreenSingle(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    testset = mobileScreenSingle(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='test')

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)

    valloader = torch.utils.data.DataLoader(
        dataset=valset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, valloader, testloader

def get_data_joint_training(args):
    base_trainset = mobileScreenSingle(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug)
    base_valset = mobileScreenSingle(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    novel_baseTestset = mobileScreenSingle(root=args.dataroot, train=False,base_sess=False,DEBUG=args.debug)
    return base_trainset, base_valset, novel_baseTestset
def get_dataloader_PretrainONE(args):
    base_trainset = mobileScreenSingle(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug,onlyNormal=True)
    base_valset = mobileScreenSingle(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val',onlyNormal=True)
    base_testset = mobileScreenSingle(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='test',onlyNormal=True)
    trainloader = torch.utils.data.DataLoader(dataset=base_trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(
        dataset=base_valset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=base_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return base_trainset,trainloader,valloader,testloader
def get_dataloader_PretrainTWO(args):
    base_trainset = mobileScreenSingle(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug)
    base_trainset.targets = [1 if i >0 else 0 for i in base_trainset.targets] #多分类更改为2分类
    base_valset = mobileScreenSingle(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    base_valset.targets = [1 if i >0 else 0 for i in base_valset.targets]
    base_testset = mobileScreenSingle(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='test')
    base_testset.targets = [1 if i >0 else 0 for i in base_testset.targets]
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(
        dataset=base_valset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=base_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return base_trainset,trainloader,valloader,testloader 
def get_new_dataloader_manyRuns(args,session): #_openness
    testset = mobileScreenSingle(root=args.dataroot, train=False,base_sess=False,DEBUG=args.debug) #base+novel
    testset.origialLegth = len(testset)
    if openness !=[]:
        openset = mobileScreenSingle(root=args.dataroot, train=True, base_sess=False,DEBUG=args.debug,args=args) #base_train属于openness类别
        testset.data.extend(openset.data)
        testset.targets.extend(openset.targets)
    return testset
    

if __name__ == '__main__':
    import argparse
    dataroot = '/media/XXX/Elements/dataset/mobile_screen/0multi_label'
    # trainset = mobileScreen(root=dataroot, train=True,base_sess=True)
    # # cls = np.unique(trainset.targets)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=2, shuffle=True, num_workers=0,
    #                                           pin_memory=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataroot', type=str, default="/media/XXX/Elements/dataset/mobile_screen/0multi_label")
    parser.add_argument('-use_back',action='store_true', default=True)

    parser.add_argument('-train_episode', type=int, default=100)
    parser.add_argument('-episode_shot', type=int, default=2)
    parser.add_argument('-episode_way', type=int, default=2)
    parser.add_argument('-episode_query', type=int, default=3)
    parser.add_argument('-test_batch_size', type=int, default=2)
    parser.add_argument('-batch_size_base', type=int, default=2)
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-debug', action='store_true',default=False)
    parser.add_argument('-shot', type=int, default=5,help='novel class')
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    args = parser.parse_args()
    trainset, trainloader, testloader = get_base_dataloader_meta(args,session=0)
    
    # trainset, trainloader, testloader = get_base_dataloader(args)

    for i, batch in enumerate(trainloader, 1):
        print("hello")


