import tensorflow as tf

'''
here implement fm in tf
y = w_0 + \sum_{i=1}^n w_ix_i + \sum_{i=1}^n \sum_{j=i+1}^n<v_i,v_j>x_ix_j
'''


def factorization_machine(x, factor_num=8):
    """
    Summary line.

    Extend description of function.

    Parameters
    ----------
    x : int
        Description of arg1
    arg2 : int
        Description of arg2

    Returns
    -------
    int
        Description of return value.
    """
    mul_of_x = tf.matmul(tf.transpose(x, perm=[0, 2, 1]), x)
    V = tf.Variable(
        tf.zeros(shape=[x.shape[-1], factor_num]), dtype=tf.float32)
    mul_of_V = tf.matmul(V, tf.transpose(V))
    order2nd = tf.reduce_sum(tf.multiply(mul_of_x, mul_of_V))

    o2sub = tf.reduce_sum(tf.multiply(V, V), axis=1)
    x2norm = tf.multiply(x, x)
    order2nd -= tf.reduce_sum(tf.multiply(x2norm, o2sub), [1, 2])
    W = tf.Variable(tf.zeros(x.shape[1:]), dtype=tf.float32)
    W0 = tf.Variable(0, dtype=tf.float32)
    prediction = W0 + tf.reduce_sum(tf.multiply(W, x), [1, 2]) + order2nd / 2
    return prediction


