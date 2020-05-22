# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:41:59 2020

@author: Administrator
"""

from utils.utils import BBoxUtility
import argparse
import cv2
import keras
import os
import json
import re
from unified_detector import Fingertips

from keras.preprocessing import image
from keras.backend.tensorflow_backend import set_session
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import optimizers
from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import PIL
from timeit import default_timer as timer
from vgg import vgg
from collections import Counter

import tensorflow as tf
from nets.ssd import SSD300 as SSD

def testdata_load(file_path):
    data = open(file_path,"r").readlines()
        
        
    for i in range(len(data)):
        data[i] = re.split("\t|,", data[i])
        data[i].pop()
    return data

class ssdT(object):
    
    def __init__(self,model,classes,input_shape):
        self.classes = classes
        self.num_class = len(classes)+1
        self.model = model
        self.input_shape = input_shape
        self.bbox_util = BBoxUtility(self.num_class)
    
        
    def image_test(self,path, inputs = None,oimg = None):
        bbox_util = BBoxUtility(2)
        if path != None:
            
            img = cv2.imread(path)
            images = img.copy()
            
            
            img = cv2.resize(img, (self.input_shape[0],self.input_shape[1]))
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            inputs = image.img_to_array(img)
            
            inputs = preprocess_input(np.array([inputs]))
        else:
            images = oimg.copy()
        preds = self.model.predict(inputs,batch_size = 1, verbose = 1)
        results = bbox_util.detection_out(preds)
        print(results)
        if len(results) > 0:
            final = []
            for each in results[0]:
                
                if each[1] < 0.4: continue
                xmin = int(each[2] * np.shape(images)[1])
                ymin = int(each[3] * np.shape(images)[0])
                xmax = int(each[4] * np.shape(images)[1])
                ymax = int(each[5] * np.shape(images)[0])
                final.append([xmin,ymin,xmax,ymax,each[1]])
            return final
        
        return None
    
    def precision(self,test_path):
        data = testdata_load(test_path)
        gnum = 0
        rnum = 0
        for eachline in data:
            res = self.image_test(eachline[0])
            gtlist = []
            temp = []
            for i in range(len(eachline)):
                if i % 5 == 0: continue
                if i % 5 == 1 and i//5 > 0: 
                    gtlist.append(temp)
                    temp = []
                temp.append(int(eachline[i]))
            gtlist.append(temp)
            print(res)
            tnum, pgnum = self.cal_iou(res, gtlist)
            gnum += pgnum
            rnum += tnum
            print("precision:",float(rnum/gnum))
        
        
    def cal_iou(self,res,gt):
        if res == None:
            return 0, len(gt)
        
        tnum = 0
        for each in gt:
            gxmin = each[0]
            gymin = each[1]
            gxmax = each[2]
            gymax = each[3]
            for one in res:
                overlap = (np.min([gxmax,one[2]]) - np.max([gxmin,one[0]])) * (np.min([gymax, one[3]]) - np.max([gymin, one[1]]))
                ares = (one[2] - one[0]) * (one[3] - one[1])
                agt = (gxmax - gxmin) * (gymax - gymin)
                wholea = agt + ares - overlap
                ratio = overlap/wholea
                if ratio > 0.7: tnum += 1
        
        return tnum, len(gt)
    

        
    def run(self, model_path, video_path = None, openposeJson = None, out_path = None, start_frame = 0, conf_threshold = 0.5, model2 = None, model3 = None):
        
        openpose_part = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                         "LShoulder", "LElbow", "LWrist", "MidHip",  "RHip", 
                         "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", 
                         "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", 
                         "RBigToe", "RSmallToe", "RHeel", "Background"]
        
        fingertips = Fingertips(weights='model_data/finmodel.h5')
        if video_path == None: return None
        video = cv2.VideoCapture(video_path)
        
        timeline = []
        labelline = []
        handStatus = []
        if out_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(out_path ,fourcc, 10.0, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))), isColor = True)
        
        vggmodel = load_model(model_path)
        if start_frame > 0:
            video.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_frame)
            
        accum_time = 0
        curr_fps = 0
        prev_time = timer()
        
        
        feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
        
        lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        color = np.random.randint(0,255,(100,3))
        num_frame = 0
        video_info = {}
        frame_info = []
        lastTime = 0
        while True:
            info, vimage = video.read()
            milliseconds = video.get(cv2.CAP_PROP_POS_MSEC)
            seconds = milliseconds / 1000
            
            video_info[str(seconds)] = []
            if not info:
                plt.figure(figsize = (100,20))
                for i in range(len(labelline)):
                    if i == 0 or i == (len(labelline) - 1):continue
                    if labelline[i] != labelline[i-1] and labelline[i] != labelline[i+1]:
                        labelline[i] = labelline[i-1]
                    
                
                for i in range(len(handStatus)):
                    if i == 0 or i == (len(handStatus) - 1):continue
                    if handStatus[i] != handStatus[i-1] and handStatus[i] != handStatus[i+1]:
                        handStatus[i] = handStatus[i-1]
                        
                #newlabelline = []
                
                for i in range(len(labelline)):
                    temp = []
                    #if i - 3 >=0: temp.append(handStatus[i-3])
                    if i - 2 >= 0: temp.append(labelline[i-2])
                    if i - 1 >= 0: temp.append(labelline[i-1])
                    temp.append(labelline[i])
                    if i + 1 < len(labelline): temp.append(labelline[i+1])
                    if i + 2 < len(labelline): temp.append(labelline[i+2])
                    #if i + 3 < len(handStatus): temp.append(handStatus[i+3])
                    labelline[i] = Counter(temp).most_common(1)[0][0]
                
                for i in range(len(handStatus)):
                    temp = []
                    #if i - 3 >=0: temp.append(handStatus[i-3])
                    if i - 2 >= 0: temp.append(handStatus[i-2])
                    if i - 1 >= 0: temp.append(handStatus[i-1])
                    temp.append(handStatus[i])
                    if i + 1 < len(handStatus): temp.append(handStatus[i+1])
                    if i + 2 < len(handStatus): temp.append(handStatus[i+2])
                    #if i + 3 < len(handStatus): temp.append(handStatus[i+3])
                    handStatus[i] = Counter(temp).most_common(1)[0][0]
                
                #np.save("labelline.npy",labelline)
                plt.plot(timeline,labelline, label = 'hand exist',color = 'r')
                plt.plot(timeline, handStatus, label = "hand status", color = 'b')
                finaltime = int(float(timeline[-1])) + 2
                plt.hlines("hand exist", 0, finaltime, color = "green", linestyles = "dashed")
                plt.hlines("hand not exist", 0, finaltime, color = "blue", linestyles = "dashed")
                plt.hlines("touch exist", 0, finaltime, color = "red", linestyles = "dashed")
                plt.hlines("no touch exist", 0, finaltime, color = "green", linestyles = "dashed")
                plt.text(finaltime, "hand exist", "hand detected at each time", fontsize = 10)
                plt.text(finaltime, "hand not exist", "hand not detected at each time",fontsize = 10)
                plt.text(finaltime, "touch exist", "hand detected and touch valid at each time", fontsize = 10)
                plt.text(finaltime, "no touch exist", "no hand or no touch valid though hand detected at each time", fontsize = 10)
                plt.xlabel("time(ms)/per frame",fontsize = 20)
                plt.ylabel("hand relative label(blue is touch validation label, red is hand detection label)", fontsize = 20)
                plt.legend()
                plt.savefig(video_path[:-4]+".jpg")
                video.release()
                if out_path: out.release()
                cv2.destroyAllWindows()
                with open(video_path[:-4]+".json","a") as outfile:
                    json.dump(video_info,outfile,ensure_ascii=False)
                    outfile.write('\n')
                print("Over")
                return
            timeline.append(round(milliseconds,2))
            input_size = (self.input_shape[0],self.input_shape[1])
            resized = cv2.resize(vimage,input_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            inputs = image.img_to_array(rgb)
            input_image = preprocess_input(np.array([inputs]))
            
            res = [[]]
            #if type(res[0]) != list: res[0] = res[0].tolist()
            if openposeJson:
                #res = [[]]
                video_file_name = os.listdir(openposeJson)
                body_info = json.load(open(openposeJson+video_file_name[num_frame],"r"))["people"]
                for h in range(len(body_info)):
                    for x in range(len(body_info[h]["pose_keypoints_2d"])):
                        if int(body_info[h]["pose_keypoints_2d"][4]) != 0:
                            if int(body_info[h]["pose_keypoints_2d"][25]) != 0:
                                distance = int((body_info[h]["pose_keypoints_2d"][25] - body_info[h]["pose_keypoints_2d"][4])/2)
                            else:
                                distance = int((np.shape(vimage)[0] - body_info[h]["pose_keypoints_2d"][4])/2)
                        else:
                            distance = 100
                        
                        if x / 3 == 4 or x / 3 == 7:
                            tres = []
                            weightsum = 0
                            xpos = int(body_info[h]["pose_keypoints_2d"][x])
                            ypos = int(body_info[h]["pose_keypoints_2d"][x+1])
                            elxpos = int(body_info[h]["pose_keypoints_2d"][x-3])
                            elypos = int(body_info[h]["pose_keypoints_2d"][x-2])
                            if xpos == 0 and ypos == 0: continue
                            
                            if elxpos >= xpos:
                                xmin = (xpos - distance) if (xpos - distance) > 0 else 0
                                xmax = (xpos + int(distance/2)) if (xpos + int(distance/2)) < np.shape(vimage)[1] else np.shape(vimage)[1]
                            else:
                                xmin = (xpos - int(distance/2)) if (xpos - int(distance/2)) > 0 else 0
                                xmax = (xpos + distance) if (xpos + distance) < np.shape(vimage)[1] else np.shape(vimage)[1]
                            
                            if elypos >= ypos:
                                ymin = (ypos - distance) if (ypos - distance) > 0 else 0
                                ymax = (ypos + int(distance/2)) if (ypos + int(distance/2)) < np.shape(vimage)[0] else np.shape(vimage)[0]
                            
                            else:
                                ymin = (ypos - int(distance/2)) if (ypos - int(distance/2)) > 0 else 0
                                ymax = (ypos + distance) if (ypos + distance) < np.shape(vimage)[0] else np.shape(vimage)[0]
                            print("distance is",distance,"box is",[xmin,ymin,xmax,ymax])
                            #cv2.rectangle(vimage,(xmin,ymin),(xmax,ymax),(255,0,0),1)
                            crop_image = vimage[ymin:ymax, xmin:xmax]
                            rgb_crop = cv2.cvtColor(cv2.resize(crop_image,input_size),cv2.COLOR_BGR2RGB)
                            input_crop = preprocess_input(np.array([image.img_to_array(rgb_crop)]))
                            if model2 == None or model3 == None:
                                if len(res) > 0:
                                    res[0].append(self.bbox_util.detection_out(self.model.predict(input_crop))[0][0])
                            else:
                                if len(combine(self,model2,model3, None, input_crop,crop_image)) > 0:
                                    #indexpro = np.array(combine(self,model2,model3, None, input_crop,crop_image))[:,1]
                                    #maxindex = np.where(indexpro == np.max(indexpro))[0][0]
                                    #each = combine(self,model2,model3, None, input_crop,crop_image)[maxindex]
                                    for each in combine(self,model2,model3, None, input_crop,crop_image):
                                        #print(each)
                                        if each[1] < conf_threshold: continue
                                            #weightsum += each[1]
                                        if each[2] <= 1 and each[3] <= 1 and each[4] <= 1 and each[5] <= 1:
                                            each[2] = int(each[2] * np.shape(crop_image)[1]) + xmin
                                            each[3] = int(each[3] * np.shape(crop_image)[0]) + ymin
                                            each[4] = int(each[4] * np.shape(crop_image)[1]) + xmin
                                            each[5] = int(each[5] * np.shape(crop_image)[0]) + ymin
                                        else:
                                            each[2] = int(each[2]) + xmin
                                            each[3] = int(each[3]) + ymin
                                            each[4] = int(each[4]) + xmin
                                            each[5] = int(each[5]) + ymin
                                        
                                        res[0].append(each)
                                        print("res is",res)
                                        
                                        
                                        #tres.append(each)
                                        
                                    """    
                                    finalbox = [1,1,0,0,0,0]
                                    for each in tres:
                                        finalbox[2] = int(finalbox[2] + each[2] * each[1]/weightsum)
                                        finalbox[3] = int(finalbox[3] + each[3] * each[1]/weightsum)
                                        finalbox[4] = int(finalbox[4] + each[4] * each[1]/weightsum)
                                        finalbox[5] = int(finalbox[5] + each[5] * each[1]/weightsum)
                                    """    
                                    
                                        
                        
                                    
                            #print(xpos, ypos)
            if len(res[0]) == 0:
                if model2 == None or model3 == None:
                
                    pred = self.model.predict(input_image)
                    
                    res = self.bbox_util.detection_out(pred)
                else:
                    #ssd ensemble learning
                    res = [combine(self,model2,model3, None, input_image,vimage)]
                            
            if len(res) > 0 and len(res[0]) > 0:
                #labelline.append("hand exist")
                
                #deal with each frame
                temp = {}
                temp["hand"] = "exist"
                temp["hand status"] = []
                temp["body part"] = []
                temp["hand position"] = []
                for each in res[0]:
                    
                    
                    if each[1] < conf_threshold: continue
                    if each[2] <= 1 and each[3] <= 1 and each[4] <= 1 and each[5] <= 1:
                        xmin = int(each[2] * np.shape(vimage)[1])
                        ymin = int(each[3] * np.shape(vimage)[0])
                        xmax = int(each[4] * np.shape(vimage)[1])
                        ymax = int(each[5] * np.shape(vimage)[0])
                    else:
                        xmin = int(each[2])
                        ymin = int(each[3])
                        xmax = int(each[4])
                        ymax = int(each[5])
                    
                    test_img = vimage[ymin:ymax,xmin:xmax]
                    
                    
                    height, width, _ = test_img.shape
                    
                    if height < 5 or width < 5:
                        finum = 0
                        continue
                       
                    else:
                        
                        temp["hand position"].append([xmin,ymin,xmax,ymax]) 
                        # gesture classification and fingertips regression
                        prob, pos = fingertips.classify(image=test_img)
                        pos = np.mean(pos, 0)
                
                        # post-processing
                        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
                        for i in range(0, len(pos), 2):
                            pos[i] = pos[i] * width + xmin
                            pos[i + 1] = pos[i + 1] * height + ymin
                        
                        # drawing
                        index = 0
                        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
                        #image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (235, 26, 158), 2)
                        finum = 0
                        for c, p in enumerate(prob):
                            if p > 0.5:
                                finum += 1
                                vimage = cv2.circle(vimage, (int(pos[index]), int(pos[index + 1])), radius=12,
                                                   color=color[c], thickness=-2)
                            index = index + 2
        
                    #edge post process
                    """
                    edges = edge(None,test_img)
                    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    test_img = cv2.subtract(test_img, edges)
                    
                    
                    test_imgr90 = cv2.flip(cv2.transpose(test_img), 1)
                    test_imgl90 = cv2.flip(cv2.transpose(test_img), 0)
                    #test_imgr90 = cv2.flip(cv2.transpose(test_img), 1)
                    
                    test_imgr90 = cv2.resize(test_imgr90,(224,224))
                    test_imgl90 = cv2.resize(test_imgl90,(224,224))
                    
                    test_imgr90 = preprocess_input(test_imgr90)
                    test_imgl90 = preprocess_input(test_imgl90)
                    
                    
                    
                    
                    test_img = cv2.resize(test_img, (224,224))
                    test_img = preprocess_input(test_img)
                    #vgg submodel detection
                    ans1 = vggmodel.predict(test_img.reshape(1,224,224,3))
                    #ans2 = vggmodel.predict(test_imgr90.reshape(1,224,224,3))
                    #ans3 = vggmodel.predict(test_imgl90.reshape(1,224,224,3))
                    pos = [ans1[0][0]]
                    """
                    
                    body_in = []
                    #for result in pos:
                    #    if result > 0.85: flag += 1
                    #print(flag)
                    cv2.rectangle(vimage, (xmin,ymin),(xmax,ymax), color = (255,0,0),thickness = 2)
                    cv2.putText(vimage, "hand", (xmin, ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
                    """
                    if flag == 0:
                        for result in pos:
                            result = result + 0.1 * (finum - 1)
                            if result > 0.7 and finum >= 2: flag += 1
                            if finum >= 3: flag += 1
                    """        
                    flag = 0
                    if flag == 0:
                        vect1 = [xmin,ymin, xmax, ymax]
                        pastTrue = 0
                        #print(frame_info)
                        for framebefore in range(len(frame_info)):
                            if frame_info[len(frame_info) - 1 - framebefore][0] == lastTime:
                                t = frame_info[len(frame_info) - 1 - framebefore]
                                vect2 = t[3:]
                                vwidth = np.min([xmax,vect2[2]]) - np.max([xmin,vect2[0]]) + 1
                                vheight = np.min([ymax,vect2[3]]) - np.max([ymin,vect2[1]]) + 1
                                
                                if vwidth < 0 or vheight < 0: continue
                                nsq = (ymax - ymin + 1) * (xmax - xmin + 1)
                                print("overlap fration:",vwidth * vheight/nsq)
                                if vwidth * vheight/nsq > 0.6:
                                    pastTrue += 1 
                                    
                            elif frame_info[len(frame_info) - 1 - framebefore][0] < lastTime:
                                break
                        
                        if pastTrue > 0 and finum >= 1:
                            flag += 1
                    
                    #flag = 1
                    
                    if openposeJson:
                        
                        video_file_name = os.listdir(openposeJson)
                        body_info = json.load(open(openposeJson+video_file_name[num_frame],"r"))["people"]
                        for h in range(len(body_info)):
                            partsplit = {"main body":[],"left hand above":[], "left hand below":[], "right hand above":[], "right hand below":[], "left leg above":[],"left leg below":[], "right leg above":[], "right leg below":[],"head":[]}
                            detail = body_info[h]["pose_keypoints_2d"]
                            if detail[51] != 0 and detail[54] != 0 and detail[4] != 0:
                                xminpos = int(np.minimum(detail[54],detail[51])) - 5
                                yminpos = int(detail[52]) - 50
                                xmaxpos = int(np.maximum(detail[51],detail[54])) + 5
                                ymaxpos = int(detail[4])
                                partsplit["head"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                                
                            if detail[6] != 0 and detail[15] != 0:
                                xminpos = int(np.minimum(detail[15],detail[6]))
                                yminpos = int(np.minimum(detail[7],detail[16]))
                                xmaxpos = int(np.maximum(detail[6],detail[15]))
                                if detail[24] != 0:
                                    ymaxpos = int(detail[25])
                                else:
                                    ymaxpos = np.shape(vimage)[0]
                                partsplit["main body"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                                
                                if detail[9] != 0:
                                    xminpos = int(np.minimum(detail[6],detail[9]))
                                    yminpos = int(np.minimum(detail[7],detail[10]))
                                    xmaxpos = int(np.maximum(detail[6],detail[9]))
                                    ymaxpos = int(np.maximum(detail[7],detail[10]))
                                    partsplit["right hand above"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                                    
                                    if detail[12] != 0:
                                        xminpos = int(np.minimum(detail[12],detail[9]))
                                        yminpos = int(np.minimum(detail[13],detail[10]))
                                        xmaxpos = int(np.maximum(detail[12],detail[9]))
                                        ymaxpos = int(np.maximum(detail[13],detail[10]))
                                        partsplit["right hand below"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                                
                                if detail[18] != 0:
                                    xminpos = int(np.minimum(detail[15],detail[18]))
                                    yminpos = int(np.minimum(detail[16],detail[19]))
                                    xmaxpos = int(np.maximum(detail[15],detail[18]))
                                    ymaxpos = int(np.maximum(detail[16],detail[19]))
                                    partsplit["left hand above"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                                    
                                    if detail[21] != 0:
                                        xminpos = int(np.minimum(detail[21],detail[18]))
                                        yminpos = int(np.minimum(detail[22],detail[19]))
                                        xmaxpos = int(np.maximum(detail[21],detail[18]))
                                        ymaxpos = int(np.maximum(detail[22],detail[19]))
                                        partsplit["left hand below"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                                        
                            if detail[27] != 0 and detail[30] != 0:
                                xminpos = int(np.minimum(detail[24],detail[30]))
                                yminpos = int(np.minimum(detail[28],detail[31]))
                                xmaxpos = int(np.maximum(detail[24],detail[30]))
                                ymaxpos = int(np.maximum(detail[28],detail[31]))
                                partsplit["right leg above"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                                    
                                if detail[33] != 0:
                                    xminpos = int(np.minimum(detail[30],detail[33]))
                                    yminpos = int(np.minimum(detail[31],detail[34]))
                                    xmaxpos = int(np.maximum(detail[30],detail[33]))
                                    ymaxpos = int(np.maximum(detail[31],detail[34]))
                                    partsplit["right leg below"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                                
                            if detail[36] != 0 and detail[39] != 0:
                                xminpos = int(np.minimum(detail[24],detail[39]))
                                yminpos = int(np.minimum(detail[37],detail[40]))
                                xmaxpos = int(np.maximum(detail[24],detail[39]))
                                ymaxpos = int(np.maximum(detail[37],detail[40]))
                                partsplit["left leg above"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                                    
                                if detail[42] != 0:
                                    xminpos = int(np.minimum(detail[39],detail[42]))
                                    yminpos = int(np.minimum(detail[40],detail[43]))
                                    xmaxpos = int(np.maximum(detail[39],detail[42]))
                                    ymaxpos = int(np.maximum(detail[40],detail[43]))
                                    partsplit["left leg below"] = [xminpos, yminpos, xmaxpos, ymaxpos]
                            
                                    
                            for x in range(len(body_info[h]["pose_keypoints_2d"])):
                                
                                if x % 3 == 0 and x / 3 != 4 and x / 3 != 7:
                                    xpos = int(body_info[h]["pose_keypoints_2d"][x])
                                    ypos = int(body_info[h]["pose_keypoints_2d"][x+1])
                                    #print(xpos, ypos)
                                    if  (xpos >= xmin and xpos <= xmax) and (ypos >= ymin and ypos <= ymax):
                                        body_in.append(openpose_part[x//3])
                                        
                            if True:
                                for keyname in partsplit.keys():
                                    if partsplit[keyname] != []:
                                        btemp = partsplit[keyname]
                                        #print(btemp)
                                        owidth = np.minimum(btemp[2], xmax) - np.maximum(xmin,btemp[0]) + 1
                                        oheight = np.minimum(btemp[3], ymax) - np.maximum(ymin,btemp[1]) + 1
                                        wholehand = (ymax - ymin + 1) * (xmax - xmin + 1)
                                        cv2.rectangle(vimage,(btemp[0],btemp[1]),(btemp[2],btemp[3]),(0,0,255),1)
                                        cv2.putText(vimage,keyname,(int((btemp[2] + btemp[0])/2)-1, btemp[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)
                                        #if keyname == "main body":
                                        #    cv2.putText(vimage,keyname,(btemp[0], btemp[3] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)
                                        #    print("main body is", btemp,"hand is",[xmin,ymin,xmax,ymax])
                                        if owidth < 0 or oheight < 0: continue
                                        oarea = owidth * oheight
                                        print("keyname is",keyname)
                                        print("flag is",flag)
                                        print("btemp is",btemp,"hand is",[xmin,ymin,xmax,ymax])
                                        print("fraction is:", oarea / wholehand)
                                        if oarea / wholehand > 0.2:
                                            body_in.append(keyname)
                                            #print("body",btemp,"hand",[xmin,ymin,xmax,ymax])
                            
                            #print((res))
                            for i in range(len(res[0])):
                                if res[0][i][1] < conf_threshold: continue
                                for j in range(i+1, len(res[0])):
                                    if res[0][j][1] < conf_threshold: continue
                                    temp1 = res[0][i]
                                    temp2 = res[0][j]
                                    width = np.min([int(temp1[4]),int(temp2[4])]) - np.max([int(temp1[2]),int(temp2[2])]) + 1
                                    height = np.min([int(temp1[5]),int(temp2[5])]) - np.max([int(temp1[3]),int(temp2[3])]) + 1
                                    if width < 0 or height < 0: continue
                                    area1 = (temp1[5] - temp1[3] + 1) * (temp1[4] - temp1[2] + 1)
                                    area2 = (temp2[5] - temp2[3] + 1) * (temp2[4] - temp2[2] + 1)
                                    overlap = width * height
                                    ratio = overlap/(area1 + area2 - overlap)
                                    if ratio > 0.6: body_in.append("hand")
                                    
                                        
                    print("body part is",body_in) 
                    frame_info.append([milliseconds,flag,finum, xmin,ymin,xmax,ymax])                   
                    if flag > 0 and len(body_in) != 0: 
                        cv2.putText(vimage, "touch", (xmax, ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
                        temp["hand status"].append("touch")
                        
                        
                    
                        
                    else: 
                        cv2.putText(vimage, "non - touch", (xmax, ymin - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
                        temp["hand status"].append("non - touch")
                    
                    if temp["hand status"][-1] == "touch":    
                        temp["body part"].append(body_in)
                    else: temp["body part"].append([])
                
                
                if len(temp["hand status"]) == 0:
                    video_info[str(seconds)].append("hand not exist")    
                    labelline.append("hand not exist")
                else:
                    video_info[str(seconds)].append(temp)
                    labelline.append("hand exist")
                
                if "touch" in temp["hand status"]:
                    
                    handStatus.append("touch exist")
                else:
                    
                    handStatus.append("no touch exist")
                
                
            else: 
                video_info[str(seconds)].append("hand not exist")    
                labelline.append("hand not exist")
                handStatus.append("no touch exist")
                        
            
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time += exec_time
            curr_fps = int(1/exec_time)
            
            num_frame += 1
            lastTime = milliseconds
            #print(curr_time, res[0])
            fps = "FPS:" + str(curr_fps)
            curr_fps = 0
            cv2.rectangle(vimage,(0,0),(50,17),(255,255,255),-1)
            cv2.putText(vimage,fps,(3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            cv2.imshow("SSD result",vimage)
            out.write(vimage)
            cv2.waitKey(1)

def combine(model1,model2,model3,img_path,inputimg = None, original_img = None):
    res1 = model1.image_test(img_path,inputs = inputimg, oimg = original_img)
    res2 = model2.image_test(img_path,inputs = inputimg, oimg = original_img)
    res3 = model3.image_test(img_path,inputs = inputimg, oimg = original_img)
    res = []
    for each in res1:
        res.append(each)
    for each in res2:
        res.append(each)
    for each in res3:
        res.append(each)
    res = np.array(res)
    if len(res) > 0:
        res = res[res[:,4].argsort()].tolist()
    
        res = nms(res)
    
    return res

def nms(res):
    maxres = []
    
    for i in range(len(res)):
        if i == 0: maxres.append(res[len(res)-1-i])
        else:
            temp = res[len(res)-1-i]
            onum = 0
            for j,each in enumerate(maxres):
                width = np.min([int(temp[2]),int(each[2])]) - np.max([int(each[0]),int(temp[0])]) + 1
                height = np.min([int(temp[3]),int(each[3])]) - np.max([int(each[1]),int(temp[1])]) + 1
                area1 = (each[3] - each[1] + 1) * (each[2] - each[0] + 1)
                area2 = (temp[3] - temp[1] + 1) * (temp[2] - temp[0] + 1)
                overlap = width * height
                ratio = overlap/(area1 + area2 - overlap)
                
                if ratio < 0.3 or width < 0 or height < 0:
                    onum += 1
                    
                else:
                    whole = maxres[j][4] + temp[4]
                    maxres[j][0] = int(maxres[j][0] * maxres[j][4]/whole + temp[0] * temp[4]/whole)
                    maxres[j][1] = int(maxres[j][1] * maxres[j][4]/whole + temp[1] * temp[4]/whole)
                    maxres[j][2] = int(maxres[j][2] * maxres[j][4]/whole + temp[2] * temp[4]/whole)
                    maxres[j][3] = int(maxres[j][3] * maxres[j][4]/whole + temp[3] * temp[4]/whole)
    
            if onum == len(maxres): maxres.append(temp)
            
    final = []
    for each in maxres:
        temp = [0]
        temp.append(each[4])
        temp.append(each[0])
        temp.append(each[1])
        temp.append(each[2])
        temp.append(each[3])
        final.append(temp)
            
    return final

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

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.compat.v1.Session(config=config))
input_shape = (300,300,3)


para_table1 = [30,60,111,162,213,264,315]
model = SSD(input_shape, 2,para_table1)
model.load_weights("model_data/ep010-loss0.738-val_loss0.679.h5",by_name=True)
video_test = ssdT(model,["hand"],input_shape)

para_table2 = [20,50, 102,150,208,255,305]
model1 = SSD(input_shape, 2,para_table2)
model1.load_weights("model_data/ep010-loss0.735-val_loss0.721.h5",by_name=True)
tmodel2 = ssdT(model1,["hand"],input_shape)


para_table3 = [35,65,120,170,220,270,320]
model2 = SSD(input_shape, 2,para_table3)
model2.load_weights("model_data/ep009-loss0.767-val_loss0.683.h5",by_name=True)
tmodel3 = ssdT(model2,["hand"],input_shape)


#res = combine(tmodel1,tmodel2,tmodel3,'VOC2007_120.jpg')


"""
res = video_test.image_test('5.jpg')
img = cv2.imread('5.jpg')
img = cv2.rectangle(img,(res[0][0],res[0][1]),(res[0][2],res[0][3]),(255,0,0),2)
cv2.imshow("dst",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

#res = video_test.image_test("t8.jpg")


####change this line to test
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--video_path', help ='original video path', type=str, default = None)
parser.add_argument('--openjson', help = 'openpose json file', type=str, default=None)
parser.add_argument('--out_path', help='output video name, should be .avi video', type = str, default=None)
parser.add_argument('--threshold', type= float, default=0.5)
args = parser.parse_args()
print(args.video_path, args.openjson, args.threshold)
video_test.run("model_data/vgghand.h5", args.video_path, openposeJson = args.openjson , out_path = args.out_path, model2 = tmodel2, model3 = tmodel3, conf_threshold = args.threshold)







