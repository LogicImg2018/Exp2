import numpy

"""
这个文件交给 柴 完成
"""

class Load_data(object):
	"""
	arg 储存图片的文件夹名称,在初始化的过程建立图像和label数组 并分别存储
	"""
	def __init__(self, arg):
		super(Load_data, self).__init__()
		self.arg = arg
		"""
		下面是需要完成的部分：获取图像和label数组
		"""
		self.train_data = []
		self.train_label = []
	
	"""
	get_data(); 用于返回 图像和label数组
	"""
	def get_data(self):
		return self.train_data, self.train_label

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


