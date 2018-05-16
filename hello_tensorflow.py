# encoding: utf-8

"""
@version: ??
@author: Mouse
@license: Apache Licence 
@contact: admin@lovexing.cn
@software: PyCharm
@file: hello_tensorflow.py
@time: 2018/5/2 20:20
"""
import tensorflow as tf


def main():
    x = tf.constant('hello world')
    with tf.Session() as sess:
        print(sess.run(x))
        print(x.name)

if __name__ == '__main__':
    main()
