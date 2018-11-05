import tensorflow as tf


def embedding_lookup(input_x, embedding_W):
    '''Generate the responsed word vector for input_x'''
    with tf.name_scope("embedding_lookup"):
        embedded_chars = tf.nn.embedding_lookup(embedding_W, input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    return embedded_chars_expanded


def embedding_layer(vocab_size, embedding_size):
    '''Create the Word Vector Matrix for the vocabulary'''
    with tf.name_scope("embedding_layer"):
        W = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            name="embedding_W")
    return W

