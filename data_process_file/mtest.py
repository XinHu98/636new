# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 10:50:49 2020

@author: Administrator
"""

import numpy as np
import os
data_path = os.listdir("hdata/")
label_path = "hlabel/"

i = 0
data = []
label = []
for j, one in enumerate(data_path):
    print(j)
    if j // 24000 != i:
        data = np.array(data)
        label = np.array(label)
        np.save("fdata/"+str(i)+".npy",data)
        np.save("flabel/"+str(i)+".npy",label)
        data = []
        label = []
        i += 1
    if i == 1:break

    data.append(np.load("hdata/"+one[:-4]+".npy").astype(np.float16))
    label.append(np.load(label_path+one[:-4]+".npy").astype(np.float16))
        
    #data = np.array(data)
    #label = np.array(label)
    
    