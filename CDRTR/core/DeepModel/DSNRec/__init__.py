import tensorflow as tf

from ..DSN.model import DSN
from ..basic import Encoder, Decoder


class DSNRec:
    def __init__(self, item_ipt, item_enc_shp,
                 item_dec_shp, user_src_ipt, user_tgt_ipt,
                 user_src_rating, src_label, tgt_label,
                 user_enc_shp, user_dec_shp, user_shr_shp):

        self.user_src_rating = user_src_rating
        self.item_ipt = item_ipt
        self.item_enc_shp = item_enc_shp
        self.item_dec_shp = item_dec_shp
        self.itemEnc = Encoder(item_ipt, item_enc_shp)
        self.itemDec = Decoder(self.itemEnc.output, item_dec_shp)
        self.itemLoss = tf.losses.mean_squared_error(self.item_ipt,
                                                     self.itemDec.output)
        self.itemHidden = self.itemEnc.output
        self.dsn_ipt = tf.concat(
            [user_src_ipt, user_tgt_ipt], axis=0)
        self.domain_label = tf.concat(
            [src_label, tgt_label], axis=0)

        self.user_enc_shp = user_enc_shp
        self.user_dec_shp = user_dec_shp
        self.user_shr_shp = user_shr_shp

        self.dsn = DSN(self.dsn_ipt, self.domain_label,
                       user_enc_shp, user_enc_shp,
                       user_dec_shp, user_dec_shp,
                       user_shr_shp)

        self.src_pred = tf.reduce_sum(
            tf.multiply(self.dsn.srcShrOut, self.itemHidden),
            axis=1)
        self.ratingLoss = tf.losses.mean_squared_error(
            self.src_pred, self.user_src_rating)

        self.totalLoss = self.ratingLoss + self.dsn.loss
