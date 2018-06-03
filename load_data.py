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
	def __init__(self, train):
		super(Load_data, self).__init__()
		self.train = train
		self.sess = tf.Session()
		self.batch_size = 4
		self.num_classes = 5
		# Get image_path and label_path

		train_image_path = 'training/*/'  # 指定训练集数据路径（根据实际情况指定训练数据集的路径）

		label_path = []

		# 打开训练数据集目录，读取全部图片，生成图片路径列表
		image_path = np.array(glob.glob(train_image_path + '*.jpg')).tolist()
		image_dir = os.listdir('training')
		for i in range(len(image_path)):

			for dir_index in range(len(image_dir)):
				if image_dir[dir_index] in image_path[i]:
					label_path.append(dir_index)

		self.num_classes = len(image_dir)
		for x in image_dir:
			print(x)
		print (self.num_classes)
		self.tr_data = ImageDataGenerator(
		    images = image_path,
		    labels = label_path,
		    batch_size = self.batch_size,
		    num_classes = self.num_classes)

		with tf.name_scope('input'):
		    # 定义迭代器
		    iterator = tf.data.Iterator.from_structure(self.tr_data.data.output_types,
		                                   self.tr_data.data.output_shapes)

		    self.training_initalize = iterator.make_initializer(self.tr_data.data)
		    # testing_initalize = iterator.make_initializer(self.test_data.data)

		    # 定义每次迭代的数据
		    self.next_batch = iterator.get_next()

	"""
	get_data(); 用于返回 图像和label数组
	"""
	def get_data(self):

		return self.train_data, self.train_label

	"""
	get_batch_data 在已有的数据中，随即挑选batch_size个图像 和 label 并返回
	"""
	def get_batch_data(self, batch_size):
		
		self.sess.run(self.training_initalize)

		train_batches_per_epoch = int(np.floor(self.tr_data.data_size / self.batch_size))

		tmp = randint(0, train_batches_per_epoch - 1)

		for step in range(train_batches_per_epoch):
			current_train_data, current_train_label = self.sess.run(self.next_batch)
			if step == tmp:
				return current_train_data, current_train_label


