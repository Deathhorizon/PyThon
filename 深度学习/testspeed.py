
# coding: utf-8

# In[1]:

import tensorflow as tf
import os
import numpy as np
import shutil
import time
from concurrent.futures import (ThreadPoolExecutor,wait)


# In[2]:
#flag is judge , fill, number
flag = "fill"
lines = tf.gfile.GFile('/Users/GF/Documents/Hexfuture/研发/Python/HTERecognition/reference/output_labels_best.txt').readlines()
uid_to_human = {}

#线程集合：这里直接声明线程数
worker_count=5;

#单个worker最大工作量，执行完后会释放内存
worker_item_max_count=200;

# task_list=[[],[],[]]
worker_list=[]

#一行一行读取数据
for uid,line in enumerate(lines) :
    #去掉换行符
    line=line.strip('\n')
    uid_to_human[uid] = line

def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


# #创建一个图来存放google训练好的模型
# with tf.gfile.FastGFile('/Users/GF/Documents/Hexfuture/研发/Python/HTERecognition/reference/output_graph_best.pb', 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     tf.import_graph_def(graph_def, name='')

# #数组的索引，平均分配到task_list中使用
# temp_index=0
# #将图片添加到集合
# for root,dirs,files in os.walk('/Users/GF/Downloads/test-稳定性/temp_choice'):
#     for file in files:
#         if temp_index is len(worker_list):
#             temp_index=0
#         worker_list[temp_index].append(os.path.join(root, file))
#         temp_index+=1

#数组的索引，平均分配到task_list中使用
#将图片添加到集合
temp_image_list =[] #临时存放的图片集合，添加到worker_list使用(非引用类型)
for root,dirs,files in os.walk('C:\Users\25119\Desktop\tech\PyThon\temp_choice'):
    for file in files:
        temp_image_list.append(os.path.join(root, file))
        if len(temp_image_list) == worker_item_max_count:
            worker_list.append(temp_image_list)
            temp_image_list=[]

#填加最后一次循环的任务
if len(temp_image_list) >0:
    worker_list.append(temp_image_list)
    temp_image_list=[]

def image_recognise(_imageList,_task_index):
    # 创建一个图来存放google训练好的模型
    with tf.gfile.FastGFile('C:/Users/25119/Desktop/tech/PyThon/深度学习/OCRServer/app/image/model/output_mobile/output_graph.pb',
                            'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        with tf.Session() as sess:
            input_x = sess.graph.get_tensor_by_name('input:0')
            softmax_tensor = sess.graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')
            index=1#执行图片序号
            for imagePath in _imageList:
                time_start = time.time()
                #print(time_start)
                #载入图片
                image = tf.gfile.FastGFile(imagePath, 'rb').read()
                image = tf.image.decode_jpeg(image, channels=3)
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
                image = tf.subtract(image, 0.5)
                image = tf.multiply(image, 2.0)

                predictions = sess.run(softmax_tensor,feed_dict={'input:0':sess.run(image)})
                predictions = np.squeeze(predictions)#把结果转为1维数据
                #排序
                top_k = predictions.argsort()[::-1]
                #print(top_k)
                cutf = top_k[0]
                human_string = id_to_string(cutf)
                condition =  human_string.find("cuterror")
                condition1 = human_string.find("CutError")

                time_end = time.time()
                print('已识别完第',_task_index,'批次的第',index,'张图片')
                print('识别结束,time cost', time_end - time_start, 's\n')
                index+=1
    return

#启动线程池
# image_recognise_threadPool=ThreadPoolExecutor(max_workers=len(worker_list))
image_recognise_threadPool=ThreadPoolExecutor(max_workers=worker_count)

#线程集合
task_list=[]
allimagecount=0

#多线程开始执行
task_index=1
for task_imagelist in worker_list:
    task = image_recognise_threadPool.submit(image_recognise,task_imagelist,task_index);
    task_list.append(task)
    allimagecount+=len(task_imagelist)
    task_index+=1

st = time.time()
wait(task_list)
et = time.time()
print('全部识别结束，',allimagecount,'张图片，耗时：',et-st,'s')

time.sleep(100000)


#task2 = executor.submit(get_html, (2))

# #遍历目录
# for root,dirs,files in os.walk('/Users/GF/Downloads/test-稳定性/temp_choice'):
#     for file in files:
#         time_start = time.time()
#         #print(time_start)
#         #载入图片
#         image = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
#         image = tf.image.decode_jpeg(image, channels=3)
#         if image.dtype != tf.float32:
#             image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#         image = tf.expand_dims(image, 0)
#         image = tf.image.resize_bilinear(image, [224, 224], align_corners=False)
#         image = tf.subtract(image, 0.5)
#         image = tf.multiply(image, 2.0)
#
#         predictions = sess.run(softmax_tensor,feed_dict={'input:0':sess.run(image)})
#         predictions = np.squeeze(predictions)#把结果转为1维数据
#
#         #打印图片路径及名称
#         image_path = os.path.join(root,file)
#         #print(image_path)
#         #显示图片
#         # img=Image.open(image_path)
#         # plt.imshow(img,cmap="gray")
#         # plt.axis('off')
#         # plt.show()
#
#         #排序
#         top_k = predictions.argsort()[::-1]
#         #print(top_k)
#         cutf = top_k[0]
#         human_string = id_to_string(cutf)
#         condition =  human_string.find("cuterror")
#         condition1 = human_string.find("CutError")
#
#         time_end = time.time()
#         print('识别结束')
#         print('time cost', time_end - time_start, 's')
#


# In[ ]:




# In[ ]:



