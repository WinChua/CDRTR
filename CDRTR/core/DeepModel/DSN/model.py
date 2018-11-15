# coding=utf8
import os

from ..basic import cnn_text
from ..basic import factorization_machine
from ..basic import embedding_lookup, embedding_layer
from ..basic import Encoder, Decoder

from CDRTR.core.LinearModel import logitRegression
from CDRTR.utils import pkdump, pkload

import tensorflow as tf


class DSN:
    def __init__(self, ipt, domain_label,
                 source_enc_shp, target_enc_shp,
                 source_dec_shp, target_dec_shp,
                 share_enc_shp, source_rating=None
                 ):
        u'''

        构造DSN模型, 这里的实现没有考虑source domain对rating的损失, 只实现了
        以下几个loss:
            * 重构损失: 领域共有编码器与私有编码器之和经过解码器重构后的损失
            * 相似损失: 领域共有编码器与私有编码器之间的相似程度, 利用两个编码器
                        输出矩阵乘积的F范数进行衡量
            * 混淆损失: 共有编码器在不同领域的输出经过一个领域分类器,通过梯度反转
                        最大化分类误差

        DSN在source domain的编码输出采用:
            self.srcShrOut

        该向量包含为共享编码器对source domain用户向量的编码输出

        Parameters
        ----------
        ipt : Tensor
        domain_label : Tensor
        source_enc_shp, target_enc_shp, source_dec_shp, target_dec_shp : list of int
        source_rating : Tensor
        '''
        self.input = ipt
        self.domain_label = domain_label
        self.source_enc_shp = source_enc_shp
        self.source_dec_shp = source_dec_shp
        self.target_enc_shp = target_enc_shp
        self.target_dec_shp = target_dec_shp
        self.source_rating = source_rating

        # 依据领域标签构造两个domain_maks用于过滤ipt
        self.source_mask = tf.equal(domain_label, 1)
        self.target_mask = tf.equal(domain_label, 0)

        # print type(self.source_mask), type(self.target_mask)
        # 利用掩码从input中过滤出分属于两个领域的输入
        self.src_ipt = tf.boolean_mask(self.input, self.source_mask)
        self.tgt_ipt = tf.boolean_mask(self.input, self.target_mask)

        # 针对两个领域的输入分别构造私有编码器
        self.srcPriEnc = Encoder(self.src_ipt, source_enc_shp)
        self.tgtPriEnc = Encoder(self.tgt_ipt, target_enc_shp)

        # 针对所有的输入构造共享编码器
        self.sharedEnc = Encoder(self.input, share_enc_shp)

        # 利用领域掩码获得两个领域数据在共享编码器上的输出
        self.srcShrOut = tf.boolean_mask(self.sharedEnc.output, self.source_mask)
        self.tgtShrOut = tf.boolean_mask(self.sharedEnc.output, self.target_mask)

        # 每个领域编解码器的隐含层输出为私有编码输出与共享编码输出之和
        self.srcHidden = self.srcPriEnc.output + self.srcShrOut
        self.tgtHidden = self.tgtPriEnc.output + self.tgtShrOut

        # 利用隐含层输出构造解码器
        self.srcDec = Decoder(self.srcHidden, source_dec_shp)
        self.tgtDec = Decoder(self.tgtHidden, target_dec_shp)

        # 重构损失
        print (self.src_ipt.shape, self.srcDec.output.shape)
        self.srcRstLoss = tf.losses.mean_squared_error(self.src_ipt, self.srcDec.output)
        self.tgtRstLoss = tf.losses.mean_squared_error(self.tgt_ipt, self.tgtDec.output)
        self.RstLoss = self.srcRstLoss + self.tgtRstLoss

        # 共享编码器输出不同领域相似损失
        self.domainProb = logitRegression(self.sharedEnc.output)
        # tf.stop_gradient使得参数的梯度不会在反向过程中传播,实现GRL梯度反转
        self.domainProb = -self.domainProb + tf.stop_gradient(self.domainProb + self.domainProb)
        # print self.domainProb.shape, self.domain_label.shape
        self.domainLoss = tf.losses.sigmoid_cross_entropy(
                tf.expand_dims(self.domain_label, -1), self.domainProb)

        # 领域私有部分与共有部分差异损失: L_loss
        self.srcDiffLoss = tf.norm(
                tf.matmul(self.srcPriEnc.output, self.srcShrOut, transpose_b=True))
        self.tgtDiffLoss = tf.norm(
                tf.matmul(self.tgtPriEnc.output, self.tgtShrOut, transpose_b=True))
        self.DiffLoss = self.srcDiffLoss + self.tgtDiffLoss

        self.loss = self.RstLoss + self.domainLoss + self.DiffLoss

        # rating prediction
        # print "share output shape:", self.srcShrOut.shape
        # print "expand_dims shape:", tf.expand_dims(self.srcShrOut, 0).shape
        # self.srcRatingPred = factorization_machine(tf.expand_dims(self.srcShrOut, -1))
        # self.ratingLoss = tf.losses.mean_squared_error(
        #         self.source_rating, tf.expand_dims(self.srcRatingPred, -1))
        # print "shape of ratingLoss is", self.ratingLoss.shape

        # self.loss = self.RstLoss + self.ratingLoss + self.domainLoss + self.DiffLoss




