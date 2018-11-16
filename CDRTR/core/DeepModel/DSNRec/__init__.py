# coding=utf8

import tensorflow as tf
import numpy as np

from ..DSN.model import DSN
from ..basic import Encoder, Decoder


class DSNRec:
    def __init__(self, item_ipt_shp, item_enc_shp,
                 item_dec_shp, user_ipt_shp, user_enc_shp,
                 user_dec_shp, user_shr_shp):

        self.tf_user_src_rating = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_item_ipt = tf.placeholder(
            shape=[None, item_ipt_shp], dtype=tf.float32)

        self.item_enc_shp = item_enc_shp
        self.item_dec_shp = item_dec_shp
        self.itemEnc = Encoder(self.tf_item_ipt, item_enc_shp)
        self.itemDec = Decoder(self.itemEnc.output, item_dec_shp)
        self.itemLoss = tf.losses.mean_squared_error(self.tf_item_ipt,
                                                     self.itemDec.output)
        self.itemHidden = self.itemEnc.output

        # 构建dsn输入, 将src跟tgt在行上进行连接, 第二个维度保持不变
        self.tf_user_src_ipt = tf.placeholder(
            shape=[None, user_ipt_shp], dtype=tf.float32)
        self.tf_user_tgt_ipt = tf.placeholder(
            shape=[None, user_ipt_shp], dtype=tf.float32)
        self.tf_dsn_ipt = tf.concat(
            [self.tf_user_src_ipt, self.tf_user_tgt_ipt], axis=0)

        # 构建领域标签
        self.tf_src_label = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_tgt_label = tf.placeholder(shape=[None], dtype=tf.float32)
        self.tf_domain_label = tf.concat(
            [self.tf_src_label, self.tf_tgt_label], axis=0)

        self.user_enc_shp = user_enc_shp
        self.user_dec_shp = user_dec_shp
        self.user_shr_shp = user_shr_shp

        self.dsn = DSN(self.tf_dsn_ipt, self.tf_domain_label,
                       user_enc_shp, user_enc_shp,
                       user_dec_shp, user_dec_shp,
                       user_shr_shp)

        self.src_pred = tf.reduce_sum(
            tf.multiply(self.dsn.srcShrOut, self.itemHidden),
            axis=1)
        self.ratingLoss = tf.losses.mean_squared_error(
            self.src_pred, self.tf_user_src_rating)

        self.totalLoss = self.ratingLoss + self.dsn.loss

        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = self.optimizer.minimize(self.totalLoss)

        self.init = tf.global_variables_initializer()

    def initSess(self, sess):
        sess.run(self.init)
        return sess

    def _buildBatch(self, batch):
        src_label = np.ones(batch["user_src_ipt"].shape[0])
        tgt_label = np.zeros(batch["user_tgt_ipt"].shape[0])
        batchData = {
            self.tf_item_ipt: batch["item_ipt"],
            self.tf_src_label: src_label,
            self.tf_tgt_label: tgt_label,
            self.tf_user_src_ipt: batch["user_src_ipt"],
            self.tf_user_src_rating: batch["user_src_rating"],
            self.tf_user_tgt_ipt: batch["user_tgt_ipt"]
        }
        return batchData

    def trainBatch(self, batch, sess):
        batchData = self._buildBatch(batch)
        _, loss = sess.run([self.train_op, self.totalLoss], feed_dict=batchData)
        return loss
