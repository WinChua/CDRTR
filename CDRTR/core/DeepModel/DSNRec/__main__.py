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
@click.option("--overlap_rate", default=1.0, help=u"两个领域交叠用户比率")
def main(data_dir, src_domain, tgt_domain, epoches, mode, overlap_rate):
    domain_name = "%s_%s" % (src_domain, tgt_domain)
    if "MultiCross" in data_dir:
        dir_name = "DSNRec/MultiCross/"
    else:
        dir_name = "DSNRec/"
    if overlap_rate != 1.0:
        dir_name = dir_name.replace("MultiCross", "ChangeOverlapRate/%s" % domain_name)
        domain_name += "_%1.2f" % overlap_rate
    runConfig = config.configs["DEBUG"](dir_name+"DSNRec_"+domain_name)
    runConfig.setConsoleLogLevel("DEBUG")
    logger = runConfig.getLogger()
    gpuConfig = runConfig.getGPUConfig()
    pre_dir = os.path.join(data_dir, "preprocess")
    dataset = DSNRecDataset.DSNRecDataset(
        os.path.join(pre_dir, "uirepresent"), os.path.join(pre_dir, "cold"),
        src_domain, tgt_domain, overlap_rate=overlap_rate)
    trainBatchGen = dataset.generateTrainBatch("user", 1000)
    # hmm, 这里的item_ipt_shp跟user_ipt_shp其实相等.
    # 因为, 不论是item还是user, 各个的向量表示都是从
    # 相同维度的句子向量表述变换而来.
    user_ipt_shp, item_ipt_shp = dataset.getUIShp()
    enc_shp = [int(user_ipt_shp*r) for r in [0.7, 0.5, 0.3]]
    dec_shp = [int(item_ipt_shp*r) for r in [0.5, 0.7, 1]]
    dsn_rec = DSNRec(
        item_ipt_shp, enc_shp, dec_shp,
        user_ipt_shp, enc_shp, dec_shp, enc_shp)
    sess = tf.Session(config=gpuConfig)
    dsn_rec.initSess(sess)
    train_writer = tf.summary.FileWriter("log/"+dir_name+domain_name+"/train", sess.graph)
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

    pkdump(test_mses, "log/"+dir_name+domain_name+"/test_mses.pk")


main()
