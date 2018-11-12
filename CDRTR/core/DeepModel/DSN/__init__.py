# coding=utf8
import os

from ..basic import cnn_text
from ..basic import factorization_machine
from ..basic import embedding_lookup, embedding_layer
from ..basic import Encoder, Decoder
from CDRTR.utils import pkdump, pkload

import tensorflow as tf


class DSN:
    def __init__(self, ipt, source_enc_shp, target_enc_shp,
                 share_enc_shp, source_dec_shp, target_dec_shp,
                 source_label, domain_label):
        if source_enc_shp[-1] != share_enc_shp[-1] and \
           target_enc_shp[-1] != share_enc_shp[-1]:
            raise Exception(u"所有编码器输出向量维度不一致")

        self.source_label = source_label
        self.domain_label = domain_label
        self.source_enc_shp = source_enc_shp
        self.target_enc_shp = target_enc_shp
        self.share_enc_shp = share_enc_shp

        # 私用, 公用编码器
        self.sourcePrivateEnc = Encoder(ipt, source_enc_shp)
        self.targetPrivateEnc = Encoder(ipt, target_enc_shp)
        self.shareEnc = Encoder(ipt, share_enc_shp)

        # 两个领域编码器的隐含层输出
        self.sourceHidden = self.sourcePrivateEnc.output + \
            self.shareEnc.output
        self.targetHidden = self.targetPrivateEnc.output + \
            self.shareEnc.output

        # 两个领域的解码器
        self.sourceDec = Decoder(self.sourceHidden, source_dec_shp)
        self.targetDec = Decoder(self.targetHidder, target_dec_shp)

        # 重构的损失
        self.sourceRstLoss = tf.losses.mean_squared_error(
                ipt,
                self.sourceDec.output)
        self.targetRstLoss = tf.losses.mean_squared_error(
                ipt,
                self.targetDec.output)
