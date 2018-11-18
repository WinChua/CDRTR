# coding=utf8
import tensorflow as tf


class _encoder(object):
    def __init__(self, ipt, enc_shp, activator):
        u'''
        Parameters
        ----------
        ipt : Tensor
            输入Tensor, shape为: [None, dims]
        enc_shp : list of int
            编/解码器每一层输出的dim
        activator : tf.nn 下的激活函数
            对于编码器, 默认使用 tf.nn.relu
            对于解码器, 默认使用 tf.nn.sigmoid
        '''
        self.input = ipt
        self.enc_lays = []
        self.enc_shp = enc_shp
        for shp in self.enc_shp:
            input = self.enc_lays[-1] if self.enc_lays else ipt
            tmp_lay = tf.layers.dense(input, shp, activator)
            self.enc_lays.append(tmp_lay)

        self.output = self.enc_lays[-1]


class Decoder(_encoder):
    def __init__(self, ipt, enc_shp):
        super(Decoder, self).__init__(ipt, enc_shp, tf.nn.sigmoid)


class Encoder(_encoder):
    def __init__(self, ipt, enc_shp):
        super(Encoder, self).__init__(ipt, enc_shp, tf.nn.sigmoid)


class AutoEncDec:
    def __init__(self, ipt, enc_shp, dec_shp):
        self.dim = ipt.shape[-1]
        if self.dim != dec_shp[-1]:
            dec_shp.append(self.dim)
        self.input = ipt
        self.encoder = Encoder(ipt, enc_shp)
        self.hidden = self.encoder.output
        self.decoder = Decoder(self.encoder.output, dec_shp)
        self.output = self.decoder.output
        self.loss = tf.losses.mean_squared_error(self.input, self.output)


class EncDec:
    def __init__(self, ipt, enc_shp, dec_shp,
                 enc_act=tf.nn.relu, dec_act=tf.nn.sigmoid):
        u'''编解码器实现

        编码层每一层的输出:
            enc_act(ipt * kernel)
        kernel 由 tf.layers.dense 创建, 列维度由enc_shp指定

        解码层类似编码层

        Parameters
        ----------
        ipt: tf.Tensor
            输入向量, shape为: [None, ipt_dims], None表示数据个数, ipt_dims表示
            每一向量的维度
        enc_shp: list of int
            每一层编码器的输出维度
            e.g. [16, 10]

        dec_shp: list of int
            每一层解码器的输出维度
            e.g. [9, 14]

        enc_act: tf的激活函数
            默认值: relu激活函数, tf.nn.relu

        dec_act: tf的激活函数
            默认值: sigmoid, tf.nn.sigmoid
        '''
        self.input = ipt
        self.enc_shp = enc_shp
        self.dec_shp = dec_shp
        self.enc_lays = []
        self.dec_lays = []

        for shp in enc_shp:
            ipt_x = self.enc_lays[-1] if self.enc_lays else ipt
            opt_h = tf.layers.dense(ipt_x, shp, enc_act)
            self.enc_lays.append(opt_h)

        for shp in dec_shp:
            ipt_x = self.dec_lays[-1] if self.dec_lays else self.enc_lays[-1]
            opt_h = tf.layers.dense(ipt_x, shp, dec_act)
            self.dec_lays.append(opt_h)

        self.hidden = self.enc_lays[-1]
        self.dims = self.input.shape[-1]
        self.output = tf.layers.dense(self.dec_lays[-1], self.dims)
        self.loss = tf.losses.mean_squared_error(self.input, self.output)

    # def getRepresent(self, ipt, sess):
    #     u'''计算输入向量的隐含层表示'''

    #     hidden = sess.run(self.hidden,
    #                       feed_dict={self.input: ipt})
    #     return hidden
