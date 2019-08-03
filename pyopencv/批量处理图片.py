#!/usr/bin/env python 
# -*- coding:utf-8 -*-


import tensorflow as tf
import os


def picread(filelist):
    """
    读取狗的图片并转换成张量
    :param filelist: 文件路f径+名字的列表
    :return: 每张图片的张量
    """
    # 1.构造文件的队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2.构造阅读器去读取图片内容（默认读取一张图片）
    reader = tf.WholeFileReader()
    key,value = reader.read(file_queue)

    # 3.对读取的图片进行解码
    image = tf.image.decode_jpeg(value)

    # 4.处理图片的大小（统一大小）
    image_resize = tf.image.resize_images(image,[200,200])

    # 注意：一定要把样本的形状固定，在批处理中要求所有数据的形状必须固定
    image_resize.set_shape([200,200,3])


    # 5.进行批处理
    image_resize_batch = tf.train.batch([image_resize],batch_size=3,num_threads=1,capacity=3)


    return   image_resize


#批处理大小，跟队列，数据的数量没有影响，只决定 这批次处理多少数据

if __name__ == "__main__":
    # 1.找到文件，放入列表  路径+名字  ->列表当中
    file_name = os.listdir("C:/Users/25119/Desktop/技术/PyThon/深度学习/OCRServer/app/image/model/test_data/temp_choice")

    filelist = [os.path.join("C:/Users/25119/Desktop/技术/PyThon/深度学习/OCRServer/app/image/model/test_data/temp_choice",file) for file in file_name ]
    image_batch= picread(filelist)

   #开启会话运行结果
    with tf.Session() as sess:
       #定义一个线程协调器
       coord = tf.train.Coordinator()

       #开启读文件的线程
       threads = tf.train.start_queue_runners(sess,coord=coord)

       #打印读取的内容
       print(sess.run([image_batch]))

       #回收子线程
       coord.request_stop()
       coord.join(threads)
