# coding=utf8
import pdb

from CDRTR.config import Config
import unittest
import tensorflow as tf
import numpy as np
from CDRTR.utils import pkdump
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
        item_enc_shp = [25, 9]
        item_dec_shp = [9, 25, item_ipt_shp]

        user_ipt_shp = 32
        usrc_ipt = np.random.randint(
            0, 9, size=record_num * user_ipt_shp).reshape((record_num, user_ipt_shp))
        utgt_ipt = np.random.randint(
            0, 9, size=2*record_num * user_ipt_shp).reshape((2*record_num, user_ipt_shp))
        usrc_rating = np.random.randint(0, 6, size=record_num)

        user_enc_shp = [25, 16, 9]
        user_dec_shp = [9, 16, 25, user_ipt_shp]
        user_shr_shp = [25, 16, 9]
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
        train_writer = tf.summary.FileWriter('log/DSNRec/Musi_Auto/train', sess.graph)
        dataset = DSNRecDataset.DSNRecDataset(
                "exam/data/preprocess/uirepresent",
                "exam/data/preprocess/cold",
                "Musi", "Auto")
        trainBatchGen = dataset.generateTrainBatch("user", 500)
        preds = []
        for epoch in range(5):
            for i in range(100):
                batchData = next(trainBatchGen)
                batch = {
                    "item_ipt": batchData["src"]["item"],
                    "user_src_ipt": batchData["src"]["user"],
                    "user_src_rating": batchData["src"]["rating"],
                    "user_tgt_ipt": batchData["tgt"]["user"]
                    }
               # for v in batch.values():
               #     print v.shape
                loss = dsn_rec.trainBatch(batch, sess)
               # print "loss of (i, epoch):(%d, %d) is %f" % (i, epoch, loss)
            # pdb.set_trace()

            pred, testloss = dsn_rec.evaluate(sess, dataset.generateTestBatch("user", 1000))
            print "the loss of test dataset is", testloss
            preds.append((epoch, pred, testloss))

        pkdump((preds, dataset.usersplit['src']['test']), "test_pred.pk")

    def test_DSNRecDataset(self):
        dataset = DSNRecDataset.DSNRecDataset(
            "exam/data/preprocess/uirepresent",
            "exam/data/preprocess/cold",
            "Auto", "Musi")
        trainBatchGen = dataset.generateTrainBatch("user", 10)
        testBatchGen = dataset.generateTestBatch("user", 10)
        for i in range(10):
            batch = next(trainBatchGen)
            # print batch
        for i in range(10):
            batch = next(testBatchGen)

