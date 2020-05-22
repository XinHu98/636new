import os

import cv2
import scipy.io as scio
import numpy as np
import json
import shutil

import re

def extract():
    dataFile = "pic/"
    files = os.listdir(dataFile)
    for m in range(len(files)):
        path = os.listdir(dataFile+files[m])
        path.sort()
        #print(path)

        data = scio.loadmat(dataFile+files[m]+"/"+path[-1])['polygons'][0]
        tinfo = {}
        for i, eachimg in enumerate(data):
            imginfo = {}
            j = 0
            for pos in eachimg:
                if np.shape(pos)[1] == 0: continue
                else:
                    j += 1
                    maxx = minx = miny = maxy = height = weight = 0
                    start = 0
                    for rpos in pos:
                        x = int(rpos[0])
                        y = int(rpos[1])
                        if start == 0:
                            minx = x
                            miny = y
                            start += 1
                        maxx = x if (x > maxx) else maxx
                        maxy = y if (y > maxy) else maxy
                        minx = x if (minx > x) else minx
                        miny = y if (miny > y) else miny

                    weight = maxx - minx
                    height = maxy - miny

                    info = [minx, miny, maxx, maxy, weight, height]
                    imginfo[j] = info

            tinfo[path[i]] = imginfo

        outfiles = open("data/"+files[m]+".json","a")
        json.dump(tinfo,outfiles,ensure_ascii=False)
        outfiles.write("\n")

#extract()

def migrate():
    filePath = "pic/"
    newPath = "images/"
    paths = os.listdir(filePath)
    prefix = [i for i in range(48)]
    for each in prefix:
        path = os.listdir(filePath+paths[each])
        for name in path:
            if name[-4:] != ".jpg": continue
            os.rename(filePath+paths[each]+"/"+name, filePath+paths[each]+"/"+str(each)+name)
            shutil.copy(filePath+paths[each]+"/"+str(each)+name, newPath+str(each)+name)

#migrate()











