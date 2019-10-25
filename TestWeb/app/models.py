from django.db import models

# Create your models here.
# ocr识别主模型
# coding: utf-8

import tensorflow as tf
import os
import numpy as np


# import shutil

class Build:
    # 识别结果集合
    labels = {}
    # 分组关系
    group_labels = {'temp_choice': {'blank', 'cancel', 'tick'},
                    'temp_judge': {'judgeblank', 'judgecancel', 'judgecross', 'judgetick'},
                    'temp_number': {'numberblank', 'numbercancel', 'numbertick'}}

    # 构造函数
    def __init__(self):
        # flag is judge , fill, number
        # flag = "judge"
        # print(flag)
        # output_labels.txt路径
        train_label_path = os.path.join(os.path.dirname(__file__), 'output_mobile/output_labels.txt')
        # output_graph.pb路径
        train_graph_path = os.path.join(os.path.dirname(__file__), 'output_mobile/output_graph.pb')

        # 加载label
        lines = tf.io.gfile.GFile(train_label_path).readlines()
        # print(lines)
        # 去掉换行符
        for uid, line in enumerate(lines):  # iterkeys、itervalues的性能也会略优于keys、values。在只需要进行迭代访问的时候优先使用iter系函数。
            line = line.strip('\n')
            self.labels[uid] = line

        # 加载深度学习库
        with tf.gfile.FastGFile(train_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    # 识别
    # is_specific：是否指定范围
    # path: 图片路径（父级名称判断范围）
    def excute(self, path, specific_name):
        with tf.Session() as sess:
            # input_x = sess.graph.get_tensor_by_name('input:0')  #大模型
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')  # 小模型
            # 载入并处理图片
            image_data = tf.gfile.FastGFile(path, 'rb').read()
            decoded_image = tf.image.decode_jpeg(image_data, channels=3)
            decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
            decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
            resize_shape = tf.stack([224, 224])
            resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
            resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
            offset_image = tf.subtract(resized_image, 127.5)
            mul_image = tf.multiply(offset_image, 1.0 / 127.5)

            predictions = sess.run(softmax_tensor, feed_dict={'input:0': sess.run(mul_image)})
            predictions = np.squeeze(predictions)  # 把结果转为1维数据
            # 排序
            top_k = predictions.argsort()[::-1]
            # 无法识别
            if self.labels[top_k[0]] == 'cuterror':
                return 'cuterror'
            # 不需要指定范围
            if specific_name:
                result = self.labels[top_k[0]]
            else:
                result = self._private_find_result(specific_name, top_k)

            return result

    def _private_find_result(self, group, indexs):
        """
        参数
        -------
            label_group:识别分组范围temp_choice、temp_judge、temp_number
            indexs:识别结果的索引list，按照准确率从大到小排序

        返回值
        -------
            识别结果，''为失败
        """
        # 获取查找的范围
        ocr_area = self.group_labels[group]
        for index in indexs:
            if self.labels[index] in ocr_area:
                return self.labels[index]
        return ''

        # #根据特定类型范围进行识别
    # def excute_specific(self, path):
    #     with tf.Session() as sess:
    #         # input_x = sess.graph.get_tensor_by_name('input:0')
    #         softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    #         # root = 'model/test_data';
    #         # 载入图片
    #         # img_path = os.path.join(root, path);
    #         image_data = tf.gfile.FastGFile(path, 'rb').read()
    #         decoded_image = tf.image.decode_jpeg(image_data, channels=3)
    #         decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    #         decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    #         resize_shape = tf.stack([224, 224])
    #         resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    #         resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    #         offset_image = tf.subtract(resized_image, 127.5)
    #         mul_image = tf.multiply(offset_image, 1.0 / 127.5)
    #
    #         predictions = sess.run(softmax_tensor, feed_dict={'input:0': sess.run(mul_image)})
    #         predictions = np.squeeze(predictions)  # 把结果转为1维数据
    #         # 排序
    #         top_k = predictions.argsort()[::-1]
    #         # print(top_k)
    #         result = self.labels[top_k[0]]
    #         return result

    # def excute2(self, path):
    #         #input_x = sess.graph.get_tensor_by_name('input:0')
    #         softmax_tensor = self.sess.graph.get_tensor_by_name('final_result:0')
    #         # root = 'model/test_data';
    #         # 载入图片
    #         # img_path = os.path.join(root, path);
    #         image_data = tf.gfile.FastGFile(path, 'rb').read()
    #         decoded_image = tf.image.decode_jpeg(image_data, channels=3)
    #         decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    #         decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    #         resize_shape = tf.stack([224, 224])
    #         resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    #         resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    #         offset_image = tf.subtract(resized_image, 127.5)
    #         mul_image = tf.multiply(offset_image, 1.0 / 127.5)
    #
    #         predictions = self.sess.run(softmax_tensor, feed_dict={'input:0': self.sess.run(mul_image)})
    #         predictions = np.squeeze(predictions)  # 把结果转为1维数据
    #         # 排序
    #         top_k = predictions.argsort()[::-1]
    #         # print(top_k)
    #         result = self.labels[top_k[0]]
    #         return result
