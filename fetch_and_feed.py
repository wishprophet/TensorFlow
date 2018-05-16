# encoding: utf-8

"""
@version: ??
@author: Mouse
@license: Apache Licence 
@contact: admin@lovexing.cn
@software: PyCharm
@file: fetch_and_feed.py
@time: 2018/5/3 21:34
"""
import tensorflow as tf
# import numpy as np

def tf_fetch():
    """
    fetch 就是一次运行多个op
    :return:
    """
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)

    # 一个加法op和一个乘法op
    add = tf.add(input2, input3)
    mul = tf.multiply(input1, add)

    with tf.Session() as sess:
        mul_result, add_result = sess.run([mul, add])
        print(mul_result, add_result)

def tf_feed():
    """
    feed
    :return:
    """
    # 创建占位符input1,input2
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        # feed 的数据已字典的形式传入
        print(sess.run(output, feed_dict={input1: [8, ], input2: [2, ]}))


if __name__ == '__main__':
    # tf_fetch()
    tf_feed()
