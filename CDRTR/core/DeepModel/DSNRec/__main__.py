# coding=utf8
import os
import click
import tensorflow as tf
from . import DSNRec
from CDRTR.dataset import DSNRecDataset
from CDRTR.utils import pkdump
from CDRTR import config


@click.command()
@click.option("--data_dir", default="data", help=u"领域数据路径")
@click.option("--src_domain", help=u"源领域名称")
@click.option("--tgt_domain", help=u"目标领域名称")
@click.option("--epoches", default=400, help=u"最大迭代次数")
@click.option("--mode", default="DEBUG", help=u"mode")
def main(data_dir, src_domain, tgt_domain, epoches, mode):
    runConfig = config.configs["DEBUG"]("DSNRec_%s_%s" % (src_domain, tgt_domain))
    runConfig.setConsoleLogLevel("DEBUG")
    logger = runConfig.getLogger()
    gpuConfig = runConfig.getGPUConfig()
    pre_dir = os.path.join(data_dir, "preprocess")
    dataset = DSNRecDataset.DSNRecDataset(
        os.path.join(pre_dir, "uirepresent"), os.path.join(pre_dir, "cold"),
        src_domain, tgt_domain)
    trainBatchGen = dataset.generateTrainBatch("user", 1000)
    item_ipt_shp, user_ipt_shp = dataset.getUIShp()
    enc_shp = [int(user_ipt_shp*r) for r in [0.7, 0.5, 0.3]]
    dec_shp = [int(item_ipt_shp*r) for r in [0.5, 0.7, 1]]
    dsn_rec = DSNRec(
        item_ipt_shp, enc_shp, dec_shp,
        user_ipt_shp, enc_shp, dec_shp, enc_shp)
    sess = tf.Session(config=gpuConfig)
    dsn_rec.initSess(sess)
    train_writer = tf.summary.FileWriter("log/DSNRec/%s_%s/train" % (src_domain, tgt_domain), sess.graph)
    test_mses = []
    for epoch in range(epoches):
        for i in range(100):
            batchData = next(trainBatchGen)
            batch = {
                "item_ipt": batchData["src"]["item"],
                "user_src_ipt": batchData["src"]["user"],
                "user_src_rating": batchData["src"]["rating"],
                "user_tgt_ipt": batchData["tgt"]["user"]
                }
            _ = dsn_rec.trainBatch(batch, sess)
        summary = dsn_rec.getSummary(sess, batch)
        train_writer.add_summary(summary, epoch)
        _, test_rmse = dsn_rec.evaluate(sess, dataset.generateTestBatch("user", 1000))
        logger.info("The test mse of epoch %d is %f.", epoch, test_rmse)
        test_mses.append(test_rmse)
    pkdump(test_mses, "log/DSNRec/%s_%s/test_mses.pk" % (src_domain, tgt_domain))


main()
