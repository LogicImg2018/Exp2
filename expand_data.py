# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import os  

def rot_sym(path, filename):
    angle = [90, 180, 270]
    img = Image.open(path + '\\' + filename); #打开文件
    img.transpose(Image.FLIP_LEFT_RIGHT).save(path + '\\' + 'sym_' + filename)  
    for ang in angle:
        img.rotate(ang).save(path + '\\' + 'rot_' + str(ang) + '_' + filename)
        img.rotate(ang).transpose(Image.FLIP_LEFT_RIGHT).save(path + '\\' + 'sym_' + str(ang) + '_' + filename)

def expand(path):
    files = os.listdir(path) #得到文件夹下的所有文件名称  
    for f in files: #遍历文件夹  
        tmp_path = os.path.join(path, f)
        if not os.path.isdir(tmp_path): #判断是否是文件夹，不是文件夹才打开  
            print('its not dir ' + f)
            rot_sym(path, f)
        else:
            expand(tmp_path)                        

def img_convert(path):
    files = os.listdir(path) #得到文件夹下的所有文件名称
    width = 277
    height = 277
    for file in files: #遍历文件夹  
        sub_path = path + '\\' + file 
        sub_files = os.listdir(sub_path)
        for img in sub_files:  #training含多个子文件夹
            img = img.resize((width, height))  
        

path = 'training' #文件夹目录  
expand(path)