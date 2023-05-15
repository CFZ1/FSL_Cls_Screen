#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:19:48 2022

@author: XXX
"""
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from dataloader.sampler import CategoriesSampler


#rolled-in---->rolled-in_scale; pitted---->pitted_surface
base_cls = ['inclusion','rolled-in','scratches']
novel_cls = ['crazing', 'patches', 'pitted']
all_cls = {'inclusion':0,'rolled-in':1,'scratches':2,'crazing':3,'patches':4,'pitted':5}

class NEUdata(Dataset):

    def __init__(self, root='/media/XXX/Elements/dataset/NEU-DET', train=True,
                 transform=None,
                 index_path=None,base_sess=None,autoaug=1,novel=False,DEBUG=False,test_mode='test'):
        
        self.base_cls = base_cls
        self.novel_cls = novel_cls        
        self.img_path = osp.join(root, 'IMAGES')
        self.data = []
        self.targets = []
        self.train = train and base_sess #----------------------05FSCF------------2
        
        if train:
            if base_sess:
                txt_path = osp.join(root,"0_fewShot_split/NEU_base_train.txt")
                self.data = [x.strip() for x in open(txt_path, 'r').readlines()]
                if DEBUG: #--------------------------------
                    print('----------------we are in debug mode----------------')
                    self.data = random.sample(self.data, 30)
            else:
                txt_path = osp.join(root,"0_fewShot_split/NEU_novel_train.txt")
                self.data = [x.strip() for x in open(txt_path, 'r').readlines()]    
        else:
            if base_sess:
                if test_mode == 'val':
                    txt_path = osp.join(root,"0_fewShot_split/NEU_base_val.txt")
                elif test_mode == 'test':
                    txt_path = osp.join(root,"0_fewShot_split/NEU_base_test.txt")
                self.data = [x.strip() for x in open(txt_path, 'r').readlines()]
                if DEBUG:#--------------------------------
                    self.data = random.sample(self.data, 6)   
            else:
                txt0_path =osp.join(root,"0_fewShot_split/NEU_base_test.txt")
                txt1_path =osp.join(root,"0_fewShot_split/NEU_novel_test.txt")
                self.data = [x.strip() for x in open(txt0_path, 'r').readlines()]
                self.data.extend([x.strip() for x in open(txt1_path, 'r').readlines()])
                if DEBUG:#--------------------------------
                    self.data = random.sample(self.data, 6)
            
        
        self.targets = [ all_cls[x.split("_")[0]] for x in self.data]
        self.data = [osp.join(self.img_path,x) for x in self.data] #img path
        
        if autoaug==0:
            
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            if train:
                 #----------------------05FSCF------------s1
#                 self.transform = transforms.Compose([
#                     transforms.RandomResizedCrop(200),
#                     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#                     transforms.RandomGrayscale(p=0.2),
#                     transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3))], p=0.5),
#                     transforms.ToTensor(),
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ])
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])   
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        #----------------------05FSCF------------3e
        if self.train:
            image = Image.open(path).convert('RGB')
            image1 = self.transform(image)
            image2 = self.transform(image)
            return image1,image2, targets
        else:
            image = self.transform(Image.open(path).convert('RGB'))
            return image, targets
        #----------------------05FSCF------------3e
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
    trainset = NEUdata(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug)
    valset = NEUdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    testset = NEUdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='test')

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
    trainset = NEUdata(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug)
    valset = NEUdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    testset = NEUdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='test')

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)

    valloader = torch.utils.data.DataLoader(
        dataset=valset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, valloader, testloader
def get_new_dataloader_manyRuns(args,session):
    trainset = NEUdata(root=args.dataroot, train=True,base_sess=False,DEBUG=args.debug) #novel    
    testset = NEUdata(root=args.dataroot, train=False,base_sess=False,DEBUG=args.debug) #base+novel
    trainset.transform = testset.transform    
    return trainset, testset

def get_data_joint_training(args):
    base_trainset = NEUdata(root=args.dataroot, train=True, base_sess=True,DEBUG=args.debug)
    base_valset = NEUdata(root=args.dataroot, train=False, base_sess=True,DEBUG=args.debug,test_mode='val')
    novelBase_testset = NEUdata(root=args.dataroot, train=False,base_sess=False,DEBUG=args.debug) #base+novel
    novel_trainset = NEUdata(root=args.dataroot, train=True,base_sess=False,DEBUG=args.debug) #novel
    return base_trainset, base_valset, novelBase_testset, novel_trainset


if __name__ == '__main__':
    
    from tqdm import tqdm
    dataroot = '/media/XXX/new/dataset/NEU-DET'
    trainset = NEUdata(root=dataroot, train=True,base_sess=True)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=2, shuffle=True, num_workers=0,
                                              pin_memory=True)
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        print("hello")
    

















