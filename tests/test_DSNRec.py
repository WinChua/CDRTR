# coding=utf8

from CDRTR.config import Config
import unittest
import tensorflow as tf
import numpy as np
from CDRTR.core.DeepModel.DSNRec import DSNRec


class DSNRecTestSuite(unittest.TestCase):
    def setUp(self):
        self.gpuConfig = Config.getGPUConfig()
        tf.logging.set_verbosity(tf.logging.ERROR)

    def test_DSNRec(self):
        ipt_shape = 100
        tf_item_ipt = tf.placeholder(
            shape=[None, 66], dtype=tf.float32)

        item_ipt = np.random.randint(0, 9, size=ipt_shape * 66).reshape((ipt_shape, 66))

        item_enc_shp = [55, 32]
        item_dec_shp = [32, 55, 66]
        tf_usrc_ipt = tf.placeholder(
            shape=[None, 100], dtype=tf.float32)
        usrc_ipt = np.random.randint(0, 9, size=ipt_shape * 100).reshape((ipt_shape, 100))
        tf_utgt_ipt = tf.placeholder(
            shape=[None, 100], dtype=tf.float32)
        utgt_ipt = np.random.randint(0, 9, size=ipt_shape * 100).reshape((ipt_shape, 100))
        tf_usrc_rating = tf.placeholder(
            shape=[None], dtype=tf.float32)
        usrc_rating = np.random.randint(0, 6, size=ipt_shape)
        tf_src_label = tf.placeholder(shape=[None], dtype=tf.int32)
        src_label = np.ones(ipt_shape)
        tf_tgt_label = tf.placeholder(shape=[None], dtype=tf.int32)
        tgt_label = np.zeros(ipt_shape)
        user_enc_shp = [77, 55, 32]
        user_dec_shp = [32, 55, 77, 100]
        user_shr_shp = [77, 55, 32]
        dsn_rec = DSNRec(
            tf_item_ipt, item_enc_shp, item_dec_shp,
            tf_usrc_ipt, tf_utgt_ipt, tf_usrc_rating,
            tf_src_label, tf_tgt_label, user_enc_shp,
            user_dec_shp, user_shr_shp)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.minimize(dsn_rec.totalLoss)
        sess = tf.Session(config=self.gpuConfig)
        init = tf.global_variables_initializer()
        sess.run(init)
        for epoch in range(50):
            _, loss = sess.run((train_op, dsn_rec.totalLoss), feed_dict={
                tf_item_ipt: item_ipt,
                tf_usrc_ipt: usrc_ipt,
                tf_utgt_ipt: utgt_ipt,
                tf_usrc_rating: usrc_rating,
                tf_src_label: src_label,
                tf_tgt_label: tgt_label,
                })
            print "epoch of ", epoch, "loss is ", loss


