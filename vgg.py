# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:59:48 2020

@author: Administrator
"""

import numpy as np
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import optimizers
from keras.applications import VGG16
import os
import cv2
import h5py
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

class vgg():
    def __init__(self, input_shape, num_classes , data_path , label_path , model_path):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.data_path = data_path
        self.label_path = label_path
        self.model_path = model_path
        self.log_path = "./log"
        self.classes = self.classname()
        
    def classname(self):
        class_dict = {"touch":1,"non-touch":2}
        return class_dict
    
    def generate_data(self,prepath = "./"):
        classes = os.listdir(prepath)
        datas = []
        labels = []
        lim = 10002
        
        #data = open(self.data_path,"a")
        #label = open(self.label_path,"a")
        for each in classes:
            if each in self.classes.keys():
                flag = 0
                file_path = os.listdir(prepath+each)
                print(each)
                for img_name in file_path:
                    if flag >= lim: continue
                    flag += 1
                    img =cv2.imread(os.path.join(prepath + each, img_name))
                    img = cv2.resize(img,(224,224))
                    img = preprocess_input(img)
                    img = img.astype(np.float16)
                    
                    datas.append(img)
                    
                    
                   # np.save(self.data_path+img_name[:-4]+".npy", img)
                    if self.classes[each] == 1:
                        labels.append([1,0])
                   #     np.save(self.label_path+img_name[:-4]+".npy",[1,0])
                    elif self.classes[each] == 2:
                        labels.append([0,1])
                   #    np.save(self.label_path+img_name[:-4]+".npy",[0,1])
       
        #datas = np.array(datas,dtype = np.float16)
        #labels = np.array(labels,dtype = np.int8)
        
        np.save(self.data_path+"data1.npy",datas)
        np.save(self.label_path+"label1.npy",labels)
        
        
    
        
        
        return True
    
    def pretrain_model(self):
        model_vgg = VGG16(include_top = False, weights = "imagenet", input_shape=(224,224,3))
        model = Flatten(name="Flatten")(model_vgg.output)
        model = Dense(self.num_classes, activation = 'softmax')(model)
        
        model_vgg = Model(inputs = model_vgg.input, outputs = model, name = 'vgg16')
        
        sgd = optimizers.SGD(lr = 0.00001, momentum = 0.9, nesterov = True)
        
        model_vgg.compile(optimizer = sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model_vgg
    
    def train(self, batch_size = 32, epoch = 10, num = 12000):
        model = self.pretrain_model()
        logs = TensorBoard(log_dir = self.log_path, write_graph = True, write_images = True)
        
        data_path = self.data_path
        label_path = self.label_path
        save_path = self.model_path
        
        data_name = os.listdir(data_path)
        
        
        for one in data_name:
            x = np.load(data_path+one)
            y = np.load(label_path+one)
            #x = np.load(data_path)
            #y = np.load(label_path)
            
            np.random.seed(200)
            np.random.shuffle(x)
            np.random.seed(200)
            np.random.shuffle(y)
            #print(logs)
            model.fit(x, y, batch_size = batch_size, epochs = epoch, verbose = 1, validation_split=0.3, callbacks = [logs])
        
        model.save(save_path)
        
    def predict(self, img_path, path = True):
        model_path = self.model_path
        model = load_model(model_path)
        
        test_img = cv2.imread(img_path)
        edges = edge(None, test_img)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        test_img = cv2.subtract(test_img, edges)
        #print(test_img)
        
        test_imgr90 = cv2.flip(cv2.transpose(test_img), 1)
        test_imgl90 = cv2.flip(cv2.transpose(test_img), 0)
        #test_imgr90 = cv2.flip(cv2.transpose(test_img), 1)
        test_img = cv2.resize(test_img, (224,224))
        test_imgr90 = cv2.resize(test_imgr90,(224,224))
        test_imgl90 = cv2.resize(test_imgl90,(224,224))
        test_img = preprocess_input(test_img)
        test_imgr90 = preprocess_input(test_imgr90)
        test_imgl90 = preprocess_input(test_imgl90)
        
        plt.imshow(test_img)
        ans1 = model.predict(test_img.reshape(1,224,224,3))
        ans2 = model.predict(test_imgr90.reshape(1,224,224,3))
        ans3 = model.predict(test_imgl90.reshape(1,224,224,3))
        fres = [ans1[0][0], ans2[0][0], ans3[0][0]]
        return fres
        
def edge(path,rimg = None):
    if path != None:
        img = cv2.imread(path)
        #img = cv2.resize(img,(300,300))
    else:
        img = rimg
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
#    cv2.imshow('dst',img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 30, 70)  # canny边缘检测   
#    cv2.imshow('thresholded',edges)
#    img90=np.rot90(edges)
#    cv2.imshow("rotate",img90)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return edges

       
