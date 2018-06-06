# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from alexnet import AlexNet 
from datetime import datetime
from load_data import Load_data
import matplotlib.pyplot as plt
from result import Result
  
data = Load_data("training1","testing")

result = Result("testing")
x_train, y_train = data.get_data();


c = x_train[25];
d = y_train[25];



x_test = data.get_test_data()
a = x_test[0];

print(np.shape(a))
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

g = rgb2gray(c)
print(d)
plt.imshow(g, cmap='Greys_r')
plt.axis('off')
plt.show()

