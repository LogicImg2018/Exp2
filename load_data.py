# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import tensorflow as tf
from random import randint
from datagenerator import ImageDataGenerator

class Load_data(object):
	"""
	train 储存训练图片的文件夹名称
	在初始化的过程建立图像和label数组 并分别存储
	"""
	def __init__(self, train,test):
		super(Load_data, self).__init__()
		self.train = train
		self.test = test
		self.sess = tf.Session()
		self.batch_size = 4
		self.num_classes = 50

	"""
	get_data(); 用于返回 图像和label数组
	"""
	def get_data(self):
		# Get image_path and label_path

		train_image_path = self.train+'/*/'  # 指定训练集数据路径（根据实际情况指定训练数据集的路径）

		label_path = []

		# 打开训练数据集目录，读取全部图片，生成图片路径列表
		image_path = np.array(glob.glob(train_image_path + '*.jpg')).tolist()
		image_dir = os.listdir(self.train)
		image_dir.sort(key= lambda x:int(x[:3]))
		for i in range(len(image_path)):

			for dir_index in range(len(image_dir)):
				if image_dir[dir_index] in image_path[i]:
					label_path.append(dir_index)

		num_classes = len(image_dir)
		batch_size = len(image_path)
		numb = 0
		for x in image_dir:
			print(numb,x)
			numb +=1
		print (num_classes)
		tr_data = ImageDataGenerator(
		    images = image_path,
		    labels = label_path,
		    batch_size = batch_size,
		    num_classes = num_classes,
		    shuffle=True)

		with tf.name_scope('input'):
		    # 定义迭代器
		    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
		                                   tr_data.data.output_shapes)

		    training_initalize = iterator.make_initializer(tr_data.data)
		    # testing_initalize = iterator.make_initializer(self.test_data.data)

		    # 定义每次迭代的数据
		    next_batch = iterator.get_next()
		self.sess.run(training_initalize)
		train_data, train_label = self.sess.run(next_batch)
		return train_data, train_label

	def get_test_data(self):
		test_image_path = self.test+'/' 

		test_label_path = []

		# 打开训练数据集目录，读取全部图片，生成图片路径列表
		image_path = np.array(glob.glob(test_image_path + '*.jpg')).tolist()
		image_path.sort(key= lambda x:int(x[16:-4]))  # read testing_1.jpg
		image_dir = os.listdir(self.test)
		image_dir.sort(key= lambda x:int(x[8:-4]))  # read testing_1.jpg
		for x in image_dir:
			print(x) 
		for i in range(len(image_path)):

			for dir_index in range(len(image_dir)):
				if image_dir[dir_index] in image_path[i]:
					test_label_path.append(dir_index)

		num_classes = len(image_dir)
		batch_size = len(image_path)
		for x in image_path:
			print(x)
		test_data = ImageDataGenerator(
		    images = image_path,
		    labels = test_label_path,
		    batch_size = batch_size,
		    num_classes = num_classes)
		with tf.name_scope('input'):
		    # 定义迭代器
		    iterator = tf.data.Iterator.from_structure(test_data.data.output_types,
		                                   test_data.data.output_shapes)

		    testing_initalize = iterator.make_initializer(test_data.data)

		    # 定义每次迭代的数据
		    next_batch = iterator.get_next()
		self.sess.run(testing_initalize)
		test_data, test_label = self.sess.run(next_batch)
		return test_data
		# return test_data.t_data

	"""
	get_batch_data 在已有的数据中，随即挑选batch_size个图像 和 label 并返回
	"""
	# def get_batch_data(self, batch_size):
		
	# 	return current_train_data, current_train_label


