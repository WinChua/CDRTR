# coding=utf8

from CDRTR.config import Config
import unittest
import tensorflow as tf
import numpy as np
from CDRTR.core.DeepModel.DSNRec import DSNRec
from CDRTR.dataset import DSNRecDataset


class DSNRecTestSuite(unittest.TestCase):
    def setUp(self):
        self.gpuConfig = Config.getGPUConfig()
        tf.logging.set_verbosity(tf.logging.ERROR)

    def test_DSNRec(self):
        record_num = 100
        item_ipt_shp = 32
        item_ipt = np.random.randint(
            0, 9, size=record_num * item_ipt_shp).reshape((record_num, item_ipt_shp))
        item_enc_shp = [55, 32]
        item_dec_shp = [32, 55, item_ipt_shp]

        user_ipt_shp = 32
        usrc_ipt = np.random.randint(
            0, 9, size=record_num * user_ipt_shp).reshape((record_num, user_ipt_shp))
        utgt_ipt = np.random.randint(
            0, 9, size=2*record_num * user_ipt_shp).reshape((2*record_num, user_ipt_shp))
        usrc_rating = np.random.randint(0, 6, size=record_num)

        user_enc_shp = [77, 55, 32]
        user_dec_shp = [32, 55, 77, user_ipt_shp]
        user_shr_shp = [77, 55, 32]
        dsn_rec = DSNRec(
            item_ipt_shp, item_enc_shp, item_dec_shp,
            user_ipt_shp, user_enc_shp, user_dec_shp, user_shr_shp)
        sess = tf.Session(config=self.gpuConfig)
        batch = {
                "item_ipt": item_ipt,
                "user_src_ipt": usrc_ipt,
                "user_src_rating": usrc_rating,
                "user_tgt_ipt": utgt_ipt
                }
        dsn_rec.initSess(sess)
        dataset = DSNRecDataset.DSNRecDataset(
                "exam/data/preprocess/uirepresent",
                "exam/data/preprocess/cold",
                "Auto", "Musi")
        trainBatchGen = dataset.generateBatch("user", 500)
        for epoch in range(50):
            for i in range(100):
                batchData = next(trainBatchGen)
                batch = {
                    "item_ipt": batchData["src"]["item"],
                    "user_src_ipt": batchData["src"]["user"],
                    "user_src_rating": batchData["src"]["rating"],
                    "user_tgt_ipt": batchData["tgt"]["user"]
                    }
                for v in batch.values():
                    print v.shape
                loss = dsn_rec.trainBatch(batch, sess)
                print "loss of (i, epoch):(%d, %d) is %f" % (i, epoch, loss)

    def test_DSNRecDataset(self):
        dataset = DSNRecDataset.DSNRecDataset(
            "exam/data/preprocess/uirepresent",
            "exam/data/preprocess/cold",
            "Auto", "Musi")
        trainBatchGen = dataset.generateBatch("user", 10)
        for i in range(10):
            batch = next(trainBatchGen)
            print batch

