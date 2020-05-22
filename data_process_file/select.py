# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:17:32 2020

@author: Administrator
"""

import numpy as np
import re
import shutil
import cv2
import string
origin_path = "Hands/Hands/"

file = open("Handinfo.txt","r")
lines = file.readlines()
for i in range(len(lines)):
    lines[i] = re.split(",", lines[i])
for j in range(1,len(lines)):
    if int(lines[j][4]) == 0 and int(lines[j][5]) == 0 and int(lines[j][8]) == 0:
        print(j)
        if "dorsal" in lines[j][6]:
            shutil.copy(origin_path+lines[j][7],"touch/"+lines[j][7][:-4]+"n.jpg")
            img = cv2.imread(origin_path+lines[j][7])
            trans_img1 = cv2.transpose( img )
            new_img1 = cv2.flip(trans_img1, 1)
            cv2.imwrite("touch/"+lines[j][7][:-4]+"r90.jpg",new_img1)
            trans_img2 = cv2.transpose( img )
            new_img2 = cv2.flip( trans_img2, 0 )
            cv2.imwrite("touch/"+lines[j][7][:-4]+"l90.jpg",new_img2)
            
            new_img3 = cv2.flip(img, 0)
            cv2.imwrite("touch/"+lines[j][7][:-4]+"180.jpg",new_img3)
            
            
        elif "palmar" in lines[j][6]:
            shutil.copy(origin_path+lines[j][7],"non-touch/"+lines[j][7][:-4]+"n.jpg")
            
            img = cv2.imread(origin_path+lines[j][7])
            trans_img1 = cv2.transpose( img )
            new_img1 = cv2.flip(trans_img1, 1)
            cv2.imwrite("non-touch/"+lines[j][7][:-4]+"r90.jpg",new_img1)
            trans_img2 = cv2.transpose( img )
            new_img2 = cv2.flip( trans_img2, 0 )
            cv2.imwrite("non-touch/"+lines[j][7][:-4]+"l90.jpg",new_img2)
            
            new_img3 = cv2.flip(img, 0)
            cv2.imwrite("non-touch/"+lines[j][7][:-4]+"180.jpg",new_img3)