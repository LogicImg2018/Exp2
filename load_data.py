# -*- coding: utf-8 -*-
import numpy

"""
这个文件交给 柴 完成
"""

class Load_data(object):
	"""
	train 储存训练图片的文件夹名称
	在初始化的过程建立图像和label数组 并分别存储
	"""
	def __init__(self, train):
		super(Load_data, self).__init__()
		self.train = train
		

		"""
		下面是需要完成的部分：
		获取: 图像[n*277*277*3] 
		     label[n*50] 每个图像对应一个向量[50]只有对应的分类结果是1 其他为0
		"""
		self.train_data = []
		self.train_label = []

	"""
	get_data(); 用于返回 图像和label数组
	"""
	def get_data(self):
		return self.train_data, self.train_label

	def get_test(self):
		return self.test
	"""
	get_batch_data 在已有的数据中，随即挑选batch_size个图像 和 label 并返回
	"""
	def get_batch_data(self,batch_size):
		


		"""
		下面是需要在已有的train_data，train_label中随即挑选batch_size个然后放在数组中返回
		"""
		current_train_data = []

		current_train_label = []

		return current_train_data,current_train_label



# 	def test(self,a,b):
			
# 		return (a+b,a-b)

# a = Load_data("data");

# b,c = a.test(5,6);


