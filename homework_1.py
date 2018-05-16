# encoding: utf-8

"""
@version: ??
@author: Mouse
@license: Apache Licence 
@contact: admin@lovexing.cn
@software: PyCharm
@file: homework_1.py
@time: 2018/5/2 22:03
"""
import tensorflow as tf
import numpy as np


def main():
    """
    使用placeholder计算Y = aX + b
    :return:
    """
    # 定义三个占位节点a,b,x
    a = tf.placeholder(tf.float32, [3, 4])
    b = tf.placeholder(tf.float32, [4, 3])
    c = tf.placeholder(tf.float32, [3, 3])
    # 计算a*b
    mul = tf.matmul(a, b)
    y = tf.add(mul, c)
    # 使用默认图

    with tf.Session() as sess:
        # 执行每一步，并喂值
        np.random.seed(5)
        ax = sess.run(mul, feed_dict={a: np.random.random((3, 4)), b: np.random.random((4, 3))})
        print(ax)
        y = sess.run(y, feed_dict={mul: ax, c: np.random.random((3, 3))})
        print(y)
        # 上下文结束自动关闭sess，资源释放，不需要Session.close()

if __name__ == '__main__':
    main()
