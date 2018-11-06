from CDRTR.dataset import Dataset
from CDRTR.utils import pkload, pkdump
from CDRTR.utils import recordTime
from CDRTR import config
import os
import glob
from operator import itemgetter

from . import SentiRec

import click
import tensorflow as tf

testEpoch = 0

@click.command()
@click.option("--dir", help=u"训练数据文件路径")
@click.option("--domain", help=u"带训练的领域")
@click.option("--filter_size", help=u"卷积核大小参数,逗号分隔")
@click.option("--filter_num", default=8, help=u"神经元个数")
@click.option("--embd_size", default=100, help=u"每个单词向量嵌入大小")
@click.option("--epoches", default=100, help=u"训练轮数")
def sentitrain(dir, domain, filter_size, filter_num, embd_size, epoches):
    runConfig = config.configs["DEBUG"]("sentitrain_%s_%s_%s" % (domain, filter_size, str(embd_size)))
    runConfig.setConsoleLogLevel("DEBUG")
    logger = runConfig.getLogger()
    gpuConfig = runConfig.getGPUConfig()
    session = tf.Session(config=gpuConfig)
    transPath = os.path.join(dir, "transform")
    data = []
    logger.info(transPath+"/*%s*" % domain)
    for d in glob.glob(transPath+"/*%s*" % domain):
        data.append(Dataset.Dataset(d))

    if data == []:
        logger.error("The data of %s is not in %s", domain, transPath)
        raise Exception

    data = data[0]
    vocabPath = os.path.join(dir, "vocab")
    vocab = pkload(os.path.join(vocabPath, "allDomain.pk"))
    vocab_size = len(vocab) + 1

    filter_size = [int(i) for i in filter_size.split(",")]

    sentc_len = data.getSentLen()
    sentirec = SentiRec(sentc_len, vocab_size, embd_size, filter_size, filter_num)
    sentirec.initSess(session)

    train_writer = tf.summary.FileWriter('log/sentitrain/%s/train' % domain, session.graph)
    test_writer = tf.summary.FileWriter('log/sentitrain/%s/test' % domain, session.graph)
    minMae = 20
    minRmse = 20
    minEpoch = epoches

    batchSize = 1000

    saver = tf.train.Saver(max_to_keep=1)

    for epoch in range(epoches):
        logger.info("Epoch %d" % epoch)

        @recordTime
        def senticEpoch():
            loss, mae, rmse = 0, 0, 0
            i = 0
            for batchData in data.getTrainBatch(batchSize, itemgetter("reviewText", "overall")):
                sentcBatch = [d[0] for d in batchData]
                ratingBatch = [d[1] for d in batchData]
                batch = {"sentc_ipt": sentcBatch,
                         "rating": ratingBatch}
                l, m, r = sentirec.trainBatch(session, batch)
                loss += l
                mae += m
                rmse += r
                i += 1
            logger.info("minMae is %f, epoch mae is %f" % (minMae, mae/i))
            logger.info("minRmse is %f, epoch rmse is %f" % (minRmse, rmse/i))
            summary = sentirec.getSummary(session, batch)
            train_writer.add_summary(summary, epoch)
            if epoch % 50 == 0:
                global testEpoch
                for testBatch in data.getTestBatch(batchSize, itemgetter("reviewText", "overall")):
                    testSB = [d[0] for d in testBatch]
                    testRB = [d[1] for d in testBatch]
                    batch = {"sentc_ipt": testSB, "rating": testRB}
                    testSummary = sentirec.getSummary(session, batch)
                    test_writer.add_summary(testSummary, testEpoch)
                    testEpoch += 1
            return mae/i, rmse/i
            return min((minMae, mae/i)), min((minRmse, rmse/i))

        mae, rmse = senticEpoch()
        if mae < minMae:
            minMae = mae
        if rmse < minRmse:
            minRmse = rmse
            minEpoch = epoch
            modelSaveDir = os.path.join(dir, "sentiModel/%s/" % domain)
            if not os.path.exists(modelSaveDir):
                os.makedirs(modelSaveDir)

            saver.save(session, os.path.join(modelSaveDir, "%s-model" % domain), global_step=epoch)

    loader = tf.train.import_meta_graph(os.path.join(modelSaveDir, "%s-model-%d.meta" % (domain, minEpoch)))
    loader.restore(session, tf.train.latest_checkpoint(modelSaveDir))

    sentiOutput = {}
    for batchData in data._getBatch(data.index, batchSize, itemgetter("reviewText", "reviewerID", "asin")):
        sentcBatch = [d[0] for d in batchData]
        reviewerIDAsin = [(d[1], d[2]) for d in batchData]
        outputVec = sentirec.outputVector(session, sentcBatch)
        sentiOutput.update(dict(zip(reviewerIDAsin, outputVec)))

    outputPath = os.path.join(dir, "sentiRecOutput", domain+".pk")
    pkdump(sentiOutput, outputPath)


sentitrain()
