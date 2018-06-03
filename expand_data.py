# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import os  

def rot_sym(path, file):
    angle = [90, 180, 270]
    img = Image.open(path+"/"+file); #打开文件
    img.transpose(Image.FLIP_LEFT_RIGHT).save(path + '/' + 'sym_' + file)  
    for ang in angle:
        img.rotate(ang).save(path + '/' + 'rot_' + str(ang) + '_' + file)
        img.rotate(ang).transpose(Image.FLIP_LEFT_RIGHT).save(path + '/' + 'sym_' + str(ang) + '_' + file)

def expand(path):
    files = os.listdir(path) #得到文件夹下的所有文件名称  
    for file in files: #遍历文件夹  
        # if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开  
        #     rot_sym(path, file)
        # else:
        sub_path = path + '/' + file 
        sub_files = os.listdir(sub_path)
        for img in sub_files:  #training含多个子文件夹
            rot_sym(sub_path, img)
                         

def img_convert(path):
    files = os.listdir(path) #得到文件夹下的所有文件名称
    width = 277
    height = 277
     for file in files: #遍历文件夹  
        sub_path = path + '/' + file 
        sub_files = os.listdir(sub_path)
        for img in sub_files:  #training含多个子文件夹
            img = img.resize((width, height))  
        

path = 'training' #文件夹目录  
expand(path)