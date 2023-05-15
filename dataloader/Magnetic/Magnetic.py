#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:19:48 2022

@author: XXX
"""
# 1.在dataloader.data_utils中注册set_up_datasets
# 2.在models/base.py中修改相应部分
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from dataloader.sampler import CategoriesSampler
import torchvision.transforms.functional as VF
# from sampler import CategoriesSampler

#rolled-in---->rolled-in_scale; pitted---->pitted_surface
base_cls = ['Free','Blowhole','Uneven','Break','Crack'] # Free-->normal
novel_cls = ['Fray']
all_cls = {'Free':0,'Blowhole':1,'Uneven':2,'Break':3,'Crack':4,'Fray':5}
Resize_w = 320
Resize_h = 320
Resize_flag = 1 

class Magneticdata(Dataset):

    def __init__(self, root='/media/XXX/Elements/dataset/Surface-Defect-Detection-master/Magnetic-Tile-Defect/', 
                 train=True,
                 transform=None,
                 index_path=None,base_sess=None,autoaug=1,novel=False,DEBUG=False,test_mode='test',args=None):
        
        self.base_cls = base_cls
        if 'Free' in base_cls:
            base_cls[base_cls.index('Free')]='normal'
        self.novel_cls = novel_cls        
        self.img_path = root
        self.data = []
        self.targets = []
        
        if train:
            if base_sess:
                txt_path = osp.join(root,"Magnetic_base_train.txt")
                self.data = [x.strip() for x in open(txt_path, 'r').readlines()]
                if DEBUG: #--------------------------------
                    print('----------------we are in debug mode----------------')
                    self.data = random.sample(self.data, 30)
            else:
                print('error:----Donot specify fixed novel_support dataset----------------') 
        else:
            if base_sess:
                if test_mode == 'val':
                    txt_path = osp.join(root,"Magnetic_base_val.txt")
                elif test_mode == 'test':
                    txt_path = osp.join(root,"Magnetic_base_test.txt")
                self.data = [x.strip() for x in open(txt_path, 'r').readlines()]
                if DEBUG:#--------------------------------
                    self.data = random.sample(self.data, 6)   
            else:
                txt0_path =osp.join(root,"Magnetic_base_test.txt")
                txt1_path =osp.join(root,"Magnetic_novel_test.txt")
                self.data = [x.strip() for x in open(txt0_path, 'r').readlines()]
                self.data.extend([x.strip() for x in open(txt1_path, 'r').readlines()])
                # if DEBUG:#--------------------------------
                #     self.data = random.sample(self.data, 6)
            
        
        self.targets = [ all_cls[x.split("/")[0].split("_")[-1]] for x in self.data]
        self.data = [osp.join(self.img_path,x) for x in self.data] #img path
        
        if Resize_flag==1:
            Resize = [transforms.Resize([Resize_w, Resize_h]),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        elif Resize_flag==2:
            Resize = [transforms.Resize([Resize_w, Resize_h]),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        elif Resize_flag==3:
            Resize = [transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            
        if autoaug==0:
            
            self.transform = transforms.Compose(Resize)
        else:
            if train:
                if (args is not None) and (args.data_augment in ['01cec']):
                    print('data_augment in 01cec')
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        # transforms.RandomVerticalFlip(),
                        transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
                        ]+Resize)
                else:
                    self.transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
                        ]+Resize)
            else:
                self.transform = transforms.Compose(Resize)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        #path = '/media/XXX/Elements/dataset/Surface-Defect-Detection-master/Magnetic-Tile-Defect/MT_Blowhole/Imgs/exp1_num_4727.jpg'
        if Resize_flag==1:
            img = Image.open(path).convert('RGB')
        elif Resize_flag==2:
            img = Image.open(path).convert('RGB')
            w,h = img.size
            if w > h:
                padding = (0,(w-h)//2,0,(w-h+1)//2) #(w, h,w, h)
                img = VF.pad(img, padding, 114, "constant")
            elif w < h:
                padding = ((h-w)//2,0,(h-w+1)//2,0)
                img = VF.pad(img, padding, 114, "constant")
        elif Resize_flag==3:
            img = Image.open(path).convert('RGB')
            w,h = img.size
            if (Resize_w/w) > (Resize_h/h):
                size = (Resize_h, Resize_h * w // h) #(h, w)
                img = VF.resize(img, size,Image.BILINEAR)
            elif (Resize_w/w) < (Resize_h/h):
                size = (Resize_w * h // w, Resize_w) #(h, w)
                img = VF.resize(img, size,Image.BILINEAR)
            elif Resize_w<w:
                size = (Resize_w * h // w, Resize_w) #(h, w)
                img = VF.resize(img, size,Image.BILINEAR)
        image = self.transform(img)
        if Resize_flag==3:
            h,w = image.shape[-2:]
            if Resize_h>h or Resize_w>w:
                padding = ((Resize_w-w)//2,(Resize_h-h)//2,(Resize_w-w+1)//2,(Resize_h-h+1)//2)
                image = VF.pad(image, padding, 0, "constant") 
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
        for novel_id in range(len(base_cls),len(base_cls)+len(novel_cls)):
            idx_list = []
            for idx,target_id in enumerate(self.targets):
                if target_id == novel_id:
                    idx_list.append(idx)
            idx_lists.append(idx_list)
        return idx_lists
#如果希望多个模型，抽取的数据顺序一样; utils.py只能保证特定模型结果可以复现
def seed_worker(use,seed):
    if use:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed) 
    else:
        print('hello')
        
def get_base_dataloader_meta(args,session):
    trainset = Magneticdata(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug,args=args)
    valset = Magneticdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    testset = Magneticdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='test')

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

def get_base_dataloader(args,session):
    trainset = Magneticdata(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug,args=args)
    valset = Magneticdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    testset = Magneticdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='test')

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)

    valloader = torch.utils.data.DataLoader(
        dataset=valset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, valloader, testloader
def get_new_dataloader_manyRuns(args,session):
    # trainset = Magneticdata(root=args.dataroot, train=True,base_sess=False,DEBUG=args.debug) #novel ---1.和NEU不同,novel的train和val来自同一处  
    testset = Magneticdata(root=args.dataroot, train=False,base_sess=False,DEBUG=args.debug) #base+novel
    # trainset.transform = testset.transform
    # return trainset, testset
    return testset

def get_data_joint_training(args):
    base_trainset = Magneticdata(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug)
    base_valset = Magneticdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    novelBase_testset = Magneticdata(root=args.dataroot, train=False,base_sess=False,DEBUG=args.debug) #base+novel
    # novel_trainset = Magneticdata(root=args.dataroot, train=True,base_sess=False,DEBUG=args.debug) #novel---2.和NEU不同,novel的train和val来自同一处  
    # return base_trainset, base_valset, novelBase_testset, novel_trainset
    return base_trainset, base_valset, novelBase_testset


if __name__ == '__main__':
    
    from tqdm import tqdm
    dataroot = '/media/XXX/Elements/dataset/Surface-Defect-Detection-master/Magnetic-Tile-Defect/'
    trainset = Magneticdata(root=dataroot, train=True,base_sess=True) #base train
    # # trainset = Magneticdata(root=dataroot, train=True,base_sess=False)
    # trainset = Magneticdata(root=dataroot, train=False,base_sess=True,test_mode='val') #base val
    # trainset = Magneticdata(root=dataroot, train=False,base_sess=True,test_mode='test') #base test
    # trainset = Magneticdata(root=dataroot, train=False,base_sess=False) #1)base test; 2)novel test(train+test)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=2, shuffle=True, num_workers=0,
                                              pin_memory=True)
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        print("hello")


