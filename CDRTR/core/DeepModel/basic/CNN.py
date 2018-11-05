import tensorflow as tf
from collections import namedtuple


def cnn_text(sentence_length, filter_sizes, num_filters,
             embedded_chars_expanded):
    # TODO : add some arguments to specify the embedding_size and the CNN layer's input.
    """
    build a cnn text preprocessor for a domain.

    Some of the code are from http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/.
    The dimensions of every variable in the network is:
        1. input_x, [the_number_of_user_in_each_batch, sentence_length]
        2. embedded_chars_expanded, [the_number_of_user_in_each_batch, sentence_length, embedding_size, 1]
        3. conv2:
            * W, [filter_size, embedding_size, 1, num_filters]
            * b, [num_filters]
        the output of conv2 is [the_number_of_user_in_each_batch, sentence_length - filter_size + 1, embedding_size - embedding_size + 1, num_filters]
        4. max_pool, [1, sentence_length - filter_size + 1, 1, 1]
        the output of max_pool is [the_number_of_user_in_each_batch, 1, 1,num_filters]
        5. tf.reshape, [the_number_of_user_in_each_batch, num_filters_total],
            which means that every user in each batch is represented as a vector of a num_filters_total size.

    Parameters
    ----------
    sentence_length : int
        specify the second dimention of input_x which represents the max length of a sentence.
    vocab_size : int
        the vocabulary size in text.
    filter_sizes : iterable
        the list which contains the filter size of every convolution kernel.
    num_filters : int
        the number of convolution kernels.
    embedded_chars_expanded : Tensor
        the input of the CNNTextProcessor.


    Returns
    -------
    input_x
        the should be use in feed_dict, {input_x: train_x}
    h_drop
        the output of the cnn text processor, should be used as the input of fm, or combined with user feature to form the input of fm.
    drop_keep_prob
        the prob of dropout layer, should be used in feed dict, {drop_keep_prob : 0.5}
    """

    dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="drop_keep_prob")
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv_maxpool_%s_%d" % (i, filter_size)):
            filter_shape = [
                filter_size, embedded_chars_expanded.shape[-2].value, 1,
                num_filters
            ]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, conv.shape[1], 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="pool")
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    return h_drop, dropout_keep_prob
