# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import os  

def expand(path):
    files = os.listdir(path) #得到文件夹下的所有文件名称  
    s = []  
    angle = [90, 180, 270]
    for file in files: #遍历文件夹  
            if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
                img = Image.open(path+"/"+file); #打开文件
                img.transpose(Image.FLIP_LEFT_RIGHT).save(path + '/' + 'sym_' + file)  
                for ang in angle:
                    img.rotate(ang).save(path + '/' + 'rot_' + str(ang) + '_' + file)
                    img.rotate(ang).transpose(Image.FLIP_LEFT_RIGHT).save(path + '/' + 'sym_' + str(ang) + '_' + file)

path = 'training' #文件夹目录  
expand(path)