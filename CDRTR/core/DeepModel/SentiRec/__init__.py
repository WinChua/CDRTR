from collections import namedtuple

from ..basic import cnn_text
from ..basic import factorization_machine
from ..basic import embedding_layer, embedding_lookup

import tensorflow as tf

batchConfig = namedtuple("batchConfig", ["sentc_ipt", "rating"])


class SentiRec(object):
    def __init__(self, sentc_len, vocab_size, embd_size,
                 filter_sizes, num_filter, drop_value=0.5):
        self.sentc_ipt = tf.placeholder(
                dtype=tf.int32,
                shape=[None, sentc_len],
                name="sentc_ipt")
        self.rating = tf.placeholder(
                dtype=tf.float32,
                shape=[None])
        self.W = embedding_layer(vocab_size, embd_size)
        self.sentc_map = embedding_lookup(self.sentc_ipt, self.W)
        self.sentc_ft, self.drop = cnn_text(sentc_len, filter_sizes,
                                            num_filter, self.sentc_map)
        self.drop_value = drop_value
        self.sentc_cnn_out = tf.expand_dims(self.sentc_ft, 1)
        self.rating_prd = factorization_machine(self.sentc_cnn_out)
        diff = self.rating_prd - self.rating
        self.loss = tf.reduce_sum(tf.pow(diff, 2))
        self.mae = tf.reduce_mean(tf.abs(diff))
        self.mse = tf.reduce_mean(tf.square(diff))
        self.rmse = tf.sqrt(self.mse)
        self.optimizer = tf.train.AdamOptimizer(0.001)
        tf.summary.scalar("mae", self.mae)
        tf.summary.scalar("rmse", self.rmse)
        self.merged = tf.summary.merge_all()
        self.train_op = self.optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def initSess(self, sess):
        sess.run(self.init)

    def _buildDict(self, batch):
        _mapper = {
                "sentc_ipt": self.sentc_ipt,
                "rating": self.rating
                }
        feed_dict = {_mapper[k]: batch[k] for k in batch}
        feed_dict[self.drop] = self.drop_value
        return feed_dict

    def trainBatch(self, sess, batch):
        _, loss, mae, rmse = sess.run(
            [self.train_op, self.loss, self.mae, self.rmse],
            feed_dict=self._buildDict(batch)
            )
        return loss, mae, rmse

    def getSummary(self, sess, batch):
        summary = sess.run(self.merged,
                           feed_dict=self._buildDict(batch)
                           )
        return summary

    def predict(self, sess, batch):
        pred = sess.run(self.rating_prd,
                        feed_dict={
                            self.sentc_ipt: batch["sentc_ipt"],
                            self.drop: self.drop_value}
                        )
        return pred

