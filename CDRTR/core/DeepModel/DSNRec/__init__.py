# coding=utf8

import tensorflow as tf
import numpy as np

from ..DSN.model import DSN
from ..basic import Encoder, Decoder


class DSNRec:
    def __init__(self, item_ipt_shp, item_enc_shp,
                 item_dec_shp, user_ipt_shp, user_enc_shp,
                 user_dec_shp, user_shr_shp):

        with tf.variable_scope("input"):
            self.tf_user_src_rating = tf.placeholder(shape=[None],
                                                     dtype=tf.float32)
            with tf.variable_scope("itemIpt"):
                self.tf_item_ipt = tf.placeholder(
                    shape=[None, item_ipt_shp], dtype=tf.float32)

            with tf.variable_scope("userIpt"):
                self.tf_user_src_ipt = tf.placeholder(
                    shape=[None, user_ipt_shp], dtype=tf.float32)
                self.tf_user_tgt_ipt = tf.placeholder(
                    shape=[None, user_ipt_shp], dtype=tf.float32)

            with tf.variable_scope("dLabelIpt"):
                # 构建领域标签
                self.tf_src_label = tf.placeholder(shape=[None],
                                                   dtype=tf.float32)
                self.tf_tgt_label = tf.placeholder(shape=[None],
                                                   dtype=tf.float32)

            self.tf_dsn_ipt = tf.concat(
                [self.tf_user_src_ipt, self.tf_user_tgt_ipt],
                axis=0, name="dsn_ipt")
            self.tf_domain_label = tf.concat(
                [self.tf_src_label, self.tf_tgt_label],
                axis=0, name="domain_label")

        with tf.variable_scope("item"):
            self.item_enc_shp = item_enc_shp
            self.item_dec_shp = item_dec_shp
            with tf.variable_scope("itemEncoder"):
                self.itemEnc = Encoder(self.tf_item_ipt, item_enc_shp)
                self.itemHidden = self.itemEnc.output
            with tf.variable_scope("itemDecoder"):
                self.itemDec = Decoder(self.itemEnc.output, item_dec_shp)

        self.user_enc_shp = user_enc_shp
        self.user_dec_shp = user_dec_shp
        self.user_shr_shp = user_shr_shp

        # 构建dsn输入, 将src跟tgt在行上进行连接, 第二个维度保持不变
        with tf.variable_scope("DSNUser"):
            self.dsn = DSN(self.tf_dsn_ipt, self.tf_domain_label,
                           user_enc_shp, user_enc_shp,
                           user_dec_shp, user_dec_shp,
                           user_shr_shp)

        with tf.variable_scope("MatrixProdRatingPred"):
            self.src_pred = tf.reduce_sum(
                tf.multiply(self.dsn.srcShrOut, self.itemHidden),
                axis=1)
            self.ratingLoss = tf.losses.mean_squared_error(
                self.src_pred, self.tf_user_src_rating)

        with tf.variable_scope("DSNLoss"):
            self.itemLoss = tf.losses.mean_squared_error(self.tf_item_ipt,
                                                         self.itemDec.output)
            self.totalLoss = self.ratingLoss + self.dsn.loss + self.itemLoss

        tf.summary.scalar("totalLoss", self.totalLoss)
        tf.summary.scalar("itemLoss", self.itemLoss)
        tf.summary.scalar("ratingMse", self.ratingLoss)
        tf.summary.scalar("dsn.srcRstLoss", self.dsn.srcRstLoss)
        tf.summary.scalar("dsn.tgtRstLoss", self.dsn.tgtRstLoss)
        tf.summary.scalar("dsn.RstLoss", self.dsn.RstLoss)
        tf.summary.scalar("dsn.domainLoss", self.dsn.domainLoss)
        tf.summary.scalar("dsn.srcDiffLoss", self.dsn.srcDiffLoss)
        tf.summary.scalar("dsn.tgtDiffLoss", self.dsn.tgtDiffLoss)
        tf.summary.scalar("dsn.DiffLoss", self.dsn.DiffLoss)
        self.merged = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = self.optimizer.minimize(self.totalLoss)

        self.init = tf.global_variables_initializer()

    def getSummary(self, sess, batch):
        batchData = self._buildBatch(batch)
        return sess.run(self.merged, feed_dict=batchData)

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

    def evaluate(self, sess, testBatches, target=None):
        u'''用测试集对target(tf的变量)进行评估

        self.ratingLoss = tf.losses.mean_squared_error(
            self.src_pred, self.tf_user_src_rating)

        Parameters
        ----------
        sess: tf.Session
            用于运行计算的session
        testBatches: generator or list of batch
            使用DSNRecDataset的generateTestBatch生成生成器
            每一次next将会生成如下字典结构:
            {"src": {"user": np.array, "item": np.array, "rating": np.array},
             "tgt": {"user": np.array, "item": np.array, "rating": np.array}}
        target: None or tf.Tensor
            sess.run方法的target, 默认值为None, 将会计算预测的评分与对应mse
            如为其他值,将会返回各个test batch对应的计算值的list

        Returns
        -------
        pred, mse: 每一份测试batch对应的预测评分以及整份test数据的mse
            如果target为None, 则返回这些数据
        results: 每一份测试batch对应的target参数中的计算值
            如果target不为None, 则返回该数据
        '''
        run_tar = [self.src_pred, self.ratingLoss] if target is None else target
        mses = []
        pred = []
        results = []
        for batchData in testBatches:
            batch = {
                "item_ipt": batchData["src"]["item"],
                "user_src_ipt": batchData["src"]["user"],
                "user_src_rating": batchData["src"]["rating"],
                "user_tgt_ipt": batchData["tgt"]["user"],
                }
            batchSize = len(batch["user_src_rating"])
            result = sess.run(run_tar,
                              feed_dict=self._buildBatch(batch))
            if target is None:
                rpred, rloss = result
                pred.append(rpred)
                mses.append((batchSize, rloss))
            else:
                results.append(result)
        if target is None:
            size, loss = 0, 0
            for bs, rl in mses:
                size += bs
                loss += rl * bs
            mse = loss / size
            return pred, mse
        else:
            return results
