#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:46:37 2022

@author: XXX
"""

import os
import random
from random import sample
import glob
random.seed(666)

base_cls = ['Free','Blowhole','Uneven','Break','Crack']  # Free-->normal, base类别再按照train:val:test = 8:1:2划分
novel_cls = ['Fray'] 
img_path = '/media/XXX/Elements/dataset/Surface-Defect-Detection-master/Magnetic-Tile-Defect/'
save_path = img_path
Subfolders = [i for i in os.listdir(img_path) if os.path.isdir(os.path.join(img_path,i))]
base_cls_paths = ['MT_'+i for i in base_cls if 'MT_'+i  in Subfolders]
novel_cls_paths = ['MT_'+i for i in novel_cls if 'MT_'+i  in Subfolders]

#--------------base---------------------------
base_train_names = []
base_val_names = []
base_test_names = []

for base_cl_path in base_cls_paths:
    base_cl_path = os.path.join(img_path,base_cl_path+'/Imgs')
    sele_names = [i.split(img_path)[-1] for i in glob.glob(base_cl_path+'/*.jpg')]
    random.shuffle(sele_names)
    train_len = int(len(sele_names)*0.7)
    val_len = int(len(sele_names)*0.1)
    test_len = int(len(sele_names)*0.2)
    base_train_names.extend(sele_names[:train_len])
    base_val_names.extend(sele_names[train_len:(train_len+val_len)])
    base_test_names.extend(sele_names[(train_len+val_len):])
    
str1 = '\n'
f=open(save_path+"Magnetic_base_train.txt","w")
f.write(str1.join(base_train_names))
f.close()    
    
str1 = '\n'
f=open(save_path+"Magnetic_base_val.txt","w")
f.write(str1.join(base_val_names))
f.close() 

str1 = '\n'
f=open(save_path+"Magnetic_base_test.txt","w")
f.write(str1.join(base_test_names))
f.close() 

#-----------------------novel-----------------------
novel_test_names = []

for novel_cl_path in novel_cls_paths:
    novel_cl_path = os.path.join(img_path,novel_cl_path+'/Imgs')
    sele_names = [i.split(img_path)[-1] for i in glob.glob(novel_cl_path+'/*.jpg')]
    novel_test_names.extend(sele_names)
# novel_test_names = []

# for novel_cl_path in novel_cls_paths:
#     novel_cl_path = os.path.join(img_path,novel_cl_path+'/Imgs')
#     sele_names = [i.split(img_path)[-1] for i in glob.glob(novel_cl_path+'/*.jpg')]
#     novel_test_names.append(sele_names)

# leng = []    
# for ind in range(len(novel_test_names)):
#     leng.append(len(novel_test_names[ind]))
# min_leng = min(leng)

# novel_test_names_new = []
# for ind in range(len(novel_test_names)):
#     if len(novel_test_names[ind]) > min_leng:
#         random.shuffle(novel_test_names[ind])
#         novel_test_names_new.extend(novel_test_names[ind][:min_leng])
#     else:
#         novel_test_names_new.extend(novel_test_names[ind])
        
str1 = '\n'
f=open(save_path+"Magnetic_novel_test.txt","w")
f.write(str1.join(novel_test_names))
f.close() 


            
        
    

  







