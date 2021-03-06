#coding=utf-8
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


# 把图片数据转化为三维矩阵
class ImageDataGenerator(object):
    def __init__(self, images, labels, batch_size, num_classes, shuffle=False):

        self.img_paths = images
        self.labels = labels
        self.num_classes = num_classes
        self.data_size = len(self.labels)
        self.pointer = 0
        self.t_data = []

        if shuffle:
            self._shuffle_lists()

        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        data = data.map(self._parse_function_train)
        
        # iterator = data.make_one_shot_iterator()
        # one_element = iterator.get_next()
        # with tf.Session() as sess:
        #     for i in range(self.data_size):
        #         x, y = sess.run(one_element)
        #         self.t_data.append(x)
        
        data = data.batch(batch_size)

        self.data = data

    # 打乱图片顺序
    def _shuffle_lists(self):
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    # 把图片生成三维数组，以及把标签转化为向量
    def _parse_function_train(self, filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, VGG_MEAN)
        img_bgr = img_centered[:, :, ::-1]
        return img_bgr, one_hot
