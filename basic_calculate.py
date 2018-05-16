# encoding: utf-8

"""
TensorFlow 基础运算
tf.Variable 变量节点
tf.constant 常量节点
tf.placeholder 占位节点
tf.zeros
tf.ones
tf.zeros_like
tf.ones_like
tf.random

@version: 0.1
@author: Mouse
@license: Apache Licence
@contact: admin@lovexing.cn
@software: PyCharm
@file: basic_calculate.py
@time: 2018/5/2 20:24
"""
import tensorflow as tf
import numpy as np


def tf_add():
    """
    两数相加
    :return:
    """
    # 定义两个常量向量a,b
    a = tf.constant(1., name='const1')
    b = tf.constant(2., name='const2')

    print(b.name, b)
    # 返回 b:0    其中b:0 表示名字为b的第0个；
    # Tensor("b:0", shape=(), dtype=float32)
    # 表示 tensor的三个特征，名字，形状，数据类型

    # 将两个向量加起来
    c = tf.add(a, b)
    # 创建会话并通过Python的上下文管理器来管理会话,执行默认计算图
    with tf.Session() as sess:
        # 使用这个创建好的会话来计算关系的结果
        result = sess.run(c)
        print(result, c.name, c)
        # 上下文结束自动关闭sess，资源释放，不需要Session.close()

    # 第二种使用Session的方式 明确的调用和关闭会话资源
    sess = tf.Session()
    result = sess.run(c)
    print(result, c.name, c)
    sess.close()


def random_num():
    """
    随机数生成
    :return:
    """
    a = np.random.rand(1)
    # 每次结果相同，由于a已经被赋值，并只赋值一次
    for i in range(5):
        print(a)

    a = tf.random_normal([1], name='random')
    # 每次的结果都不一样，因为每次都是重新去运行该a节点
    with tf.Session() as sess:
        for i in range(5):
            sess.run(a)
            # 可以通过tensor的eval方法获得tensor的值
            print(a.name, a.eval(), a)


def use_placeholder():
    """
    使用placeholder计算Y = aX + b
    :return:
    """
    # 定义三个占位节点a,b,x
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)
    x = tf.placeholder(tf.int16)
    # 计算a*X
    mul = tf.multiply(a, x)
    y = tf.add(mul, b)
    # 使用默认图
    with tf.Session() as sess:
        # 执行每一步，并喂值
        ax = sess.run(mul, feed_dict={a: 2, x: 3})
        print(ax)
        y = sess.run(y, feed_dict={mul: ax, b: 3})
        print(y)
        # 上下文结束自动关闭sess，资源释放，不需要Session.close()


def define_graph():
    """
    自定义计算图
    :return:
    """
    g = tf.Graph()
    with g.as_default():
        a = tf.constant(2, name='a')
        b = tf.constant(3, name='b')
        x = tf.add(a, b, name='add')
        print(x.name, x)
        # 创建回话计算结果 x 是一个局部变量
        with tf.Session() as sess:
            result = sess.run(x)
            print(result, x.name, x)


def variable_init():
    """
    使用TensorFlow变量，初始化
    :return:
    """
    x = tf.Variable([1, 2])
    a = tf.constant([3, 3])
    # 增加一个减法op
    sub = tf.subtract(x, a)
    # 增加一个加法op
    add = tf.add(x, sub)
    # 使用变量需要初始化
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 会话中先初始化变量
        sess.run(init)
        print(sess.run(sub))
        print(sess.run(add))


def iadd():
    """
    自增
    :return:
    """
    # 变量state
    state = tf.Variable(0, name='counter')
    new_value = tf.add(state, 1)
    # tensorflow中赋值需要使用assign
    update = tf.assign(state, new_value)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 初始化变量
        sess.run(init)
        # 初始状态
        print(sess.run(state))
        for i in range(5):
            # 自增
            sess.run(update)
            # 查看值
            print(sess.run(state))

if __name__ == '__main__':
    # use_placeholder()
    # tf_add()
    # define_graph()
    # random_num()
    # variable_init()
    iadd()
