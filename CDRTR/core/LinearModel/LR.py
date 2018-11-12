import tensorflow as tf


def logitRegression(ipt):
    W = tf.Variable(tf.zeros([ipt.shape[-1], 1]), dtype=tf.float32)
    bias = tf.Variable(0.0, dtype=tf.float32)
    y = tf.nn.sigmoid(tf.matmul(ipt, W) + bias)
    return y
