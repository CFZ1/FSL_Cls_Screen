#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:46:37 2022

@author: XXX
"""

import os
import random
from random import sample
random.seed(666)

base_cls = ['inclusion','rolled-in','scratches'] #rolled-in---->rolled-in_scale
novel_cls = ['crazing','patches','pitted'] #pitted---->pitted_surface
img_path = '/media/XXX/Elements/dataset/NEU-DET/IMAGES'
save_path = '/media/XXX/Elements/dataset/NEU-DET/0_fewShot_split/'
all_names = os.listdir(img_path)
random.shuffle(all_names)

base_names = [x for x in all_names if x.split("_")[0] in base_cls]
novel_names = list(set(all_names) - set(base_names)) 

#--------------base---------------------------
base_train_names = []
base_val_names = []
base_test_names = []
train_len = int(300*0.7)
val_len = int(300*0.1)
test_len = int(300*0.2)

for base_cl in base_cls:
    sele_names = [x for x in base_names if x.split("_")[0] in base_cl]
    base_train_names.extend(sele_names[:train_len])
    base_val_names.extend(sele_names[train_len:(train_len+val_len)])
    base_test_names.extend(sele_names[(train_len+val_len):])
    
str1 = '\n'
f=open(save_path+"NEU_base_train.txt","w")
f.write(str1.join(base_train_names))
f.close()    
    
str1 = '\n'
f=open(save_path+"NEU_base_val.txt","w")
f.write(str1.join(base_val_names))
f.close() 

str1 = '\n'
f=open(save_path+"NEU_base_test.txt","w")
f.write(str1.join(base_test_names))
f.close() 

#-----------------------novel-----------------------
shots =[1,2,5] 
episode = 100

novel_train_names = []
novel_test_names = []
train_len = int(300*0.8)

for novel_cl in novel_cls:
    sele_names = [x for x in novel_names if x.split("_")[0] in novel_cl]
    novel_train_names.extend(sele_names[:train_len])
    novel_test_names.extend(sele_names[train_len:])

str1 = '\n'
f=open(save_path+"NEU_novel_train.txt","w")
f.write(str1.join(novel_train_names))
f.close() 

str1 = '\n'
f=open(save_path+"NEU_novel_test.txt","w")
f.write(str1.join(novel_test_names))
f.close() 


            
        
    

  







