# coding=utf8

from CDRTR.config import Config
import unittest
import tensorflow as tf
import numpy as np
from CDRTR.core.DeepModel.DSN.model import DSN

class DSNTestSuite(unittest.TestCase):
    def setUp(self):
        self.gpuConfig = Config.getGPUConfig()
        tf.logging.set_verbosity(tf.logging.ERROR)

    def test_all(self):
        tf_ipt_x = tf.placeholder(shape=[None, 100], dtype=tf.float32)
        ipt_x = np.random.randint(0, 9, size=(10, 100))
        tf_domain_label = tf.placeholder(shape=[None], dtype=tf.int32)
        domain_label = np.random.randint(0, 2, size=(10,))
        enc_shp = [77, 55, 32]
        dec_shp = [32, 55, 77, 100]
        # 使用的时候需要注意, tf_rating, tf_ipt_itemx都是source domain的,
        # 需要保持shape一致
        tf_rating = tf.placeholder(shape=[None], dtype=tf.float32)
        rating = np.random.randint(1, 6, size=(sum(domain_label),))
        dsn = DSN(tf_ipt_x, tf_domain_label,
                  enc_shp, enc_shp,
                  dec_shp, dec_shp,
                  enc_shp, tf_rating)
        tf_ipt_itemx = tf.placeholder(shape=[None, 32], dtype=tf.float32)
        ipt_itemx = np.random.randint(0, 9, size=(sum(domain_label), 32))
        print "srcShrOut shape is ", dsn.srcShrOut.shape, tf_ipt_itemx.shape
        tf_pred_rating = tf.reduce_sum(
                tf.multiply(dsn.srcShrOut, tf_ipt_itemx),
                axis=1)
        print tf_rating.shape, tf_pred_rating.shape
        loss = tf.losses.mean_squared_error(tf_rating, tf_pred_rating)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        print sess.run(loss, feed_dict={
            tf_ipt_x: ipt_x,
            tf_rating: rating,
            tf_ipt_itemx: ipt_itemx,
            tf_domain_label: domain_label})
