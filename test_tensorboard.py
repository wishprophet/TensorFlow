# encoding: utf-8

"""
@version: ??
@author: Mouse
@license: Apache Licence
@contact: admin@lovexing.cn
@software: PyCharm
@file: tensorboard.py
@time: 2018/5/10 9:36
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    # 载入数据
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 每个批次大小
    batch_size = 50
    # 计算批次个数
    n_batch = mnist.train.num_examples // batch_size
    with tf.name_scope('input'):
        # 定义两个placeholder
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    # dropout参数
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.Variable(0.001, dtype=tf.float32)

    with tf.name_scope('layer'):
        # 创建一个简单的神经网络
        # 隐藏层
        W1 = tf.Variable(tf.truncated_normal([784, 600], stddev=0.1))
        b1 = tf.Variable(tf.zeros([600]) + 0.1)
        L1 = tf.nn.tanh(tf.matmul(x, W1)+b1)
        L1_drop = tf.nn.dropout(L1, keep_prob)

        W2 = tf.Variable(tf.truncated_normal([600, 400], stddev=0.1))
        b2 = tf.Variable(tf.zeros([400]) + 0.1)
        L2 = tf.nn.tanh(tf.matmul(L1_drop, W2)+b2)
        L2_drop = tf.nn.dropout(L2, keep_prob)

        W3 = tf.Variable(tf.truncated_normal([400, 200], stddev=0.1))
        b3 = tf.Variable(tf.zeros([200]) + 0.1)
        L3 = tf.nn.tanh(tf.matmul(L2_drop, W3)+b3)
        L3_drop = tf.nn.dropout(L3, keep_prob)

        # 输出层
        # 权值初始化截断的正态分布标准差为0.1
        # 偏执值初始化 0+0.1
        W = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
        b = tf.Variable(tf.zeros([10]) + 0.1)
        prediction = tf.nn.softmax(tf.matmul(L3_drop, W)+b)

    # 二次代价函数
    # loss = tf.reduce_mean(tf.square(y-prediction))
    # 交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # Adam
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # 其他优化器
    # train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)
    # 初始化
    init = tf.global_variables_initializer()

    # 结果存放在一个布尔型列表中
    # argmax返回一维张量中最大的值所在的位置，mnist中label：([1,0,0,0,0,0,0,0,0,0,0])，
    # agrmax返回的就是1所在的位置，如果预测值与所给的标签集相同，表示成功识别数字，返回值为1，反之为0
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # cast转换数据类型，Bool-Float，然后计算平均值就是准确度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(init)
        # 在当前目录下logs目录，存储图结构
        writer = tf.summary.FileWriter('logs/', sess.graph)
        for epoch in range(1):
            sess.run(tf.assign(learning_rate, 0.001*(0.95**epoch)))
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.9})
            lr = sess.run(learning_rate)
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 0.9})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 0.9})
            # train_acc 和 test_acc差的比较多说明过拟合
            print('epoch ' + str(epoch) + ' lr:' + str(lr) + ' test_acc:' + str(test_acc)+' train_acc:' + str(train_acc))


if __name__ == '__main__':
    main()
