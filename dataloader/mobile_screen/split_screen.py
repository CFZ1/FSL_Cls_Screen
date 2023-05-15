#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 09:27:40 2022

@author: XXX
"""
import numpy as np
import pandas as pd
# import os
import warnings
import torch
import copy
import os
import csv

base_cls = ['pinhole','scratch','tin']
novel_cls =['bubble']
csv_row1 = ['img_names',"bubble","pinhole","scratch","tin",] #本来是'bubble',"pinhole","scratch","tin"
train_ann_file = "/media/XXX/new/dataset/mobile_screen/0multi_label/slice_img/train_singleLabel.csv"
val_ann_file = "/media/XXX/new/dataset/mobile_screen/0multi_label/slice_img/val_singleLabel.csv"
save_path = "/media/XXX/new/dataset/mobile_screen/0multi_label/split_FSCIL_cls/"

def split_baseNovel(origin_path):
    
    train_data = pd.read_csv(origin_path) 
    # #bubble移动到最后一列
    # bubble = train_data.pop('bubble')
    # train_data.insert(loc=train_data.shape[1], column='bubble', value=bubble, allow_duplicates=False)
    
    train_data = np.array(train_data).tolist()
    base_train = []
    novel_train = []
    for data_row in train_data:
        base_flag = True
        for novel_cl in novel_cls:
            if novel_cl in data_row[0]: #e.g. 'bubble' in 'bubble/20180713113334_2_819_819.jpg'
                novel_train.append(copy.deepcopy(data_row))
                base_flag = False
        if base_flag:
            base_train.append(copy.deepcopy(data_row))
    return base_train,novel_train

def save_csv(content,file_name):
    with open(os.path.join(save_path,file_name),"w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_row1)
        for num in range(len(content)):
            wo = content[num]
            writer.writerow(wo)

base_train,novel_train = split_baseNovel(train_ann_file)
base_val,novel_val = split_baseNovel(val_ann_file)

save_csv(base_train,"base_train.csv")
save_csv(novel_train,"novel_train.csv")
save_csv(base_val,"base_val.csv")
save_csv(novel_val,"novel_val.csv")

        

        
        
    
