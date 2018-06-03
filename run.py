#! usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np  
import pandas as pd  
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

# from keras.utils import plot_model

from load_data import Load_data

from result import Result

data = Load_data("train")

result = Result("test")



batch_size = 128

num_classes = data.num_classes

epochs = 30  #设置迭代次数

img_rows, img_cols = 227, 227

# train = pd.read_csv('train.csv')
# test = pd.read_csv("test.csv")




x_test,x_test_label = data.get_batch_data (1000)

x_train,y_train  = data.get_batch_data (1000)


input_shape = (img_rows, img_cols, 3)




mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

model = Sequential()

model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
model.add(Flatten())  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes,activation='softmax'))  
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy']) 



# model.compile(
#               # loss=keras.losses.categorical_crossentropy,   
#               loss = keras.losses.kullback_leibler_divergence,#交叉熵
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

model = load_model('bvlc_alexnet.npy') 

predictions = model.predict_classes(x_test, verbose=0)

submissions = pd.DataFrame(
    {"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions})
submissions.to_csv("result3.csv", index=False, header=True)

from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))

# 输出文件为result3.csv 在当前目录下。
# plot_model(model,to_file='model.png')
# 生成网络结构示意图



















