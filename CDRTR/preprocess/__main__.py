# -*- coding: utf8 -*-

import os
import sys
import click

from .utils import buildVoca, transReview
from ..utils import pkdump, recordTime
from .. import config
from .cold import generateColdUser


@click.group()
def cli():
    pass


@cli.command()
@click.option("--mode", type=click.Choice(
    ["DEBUG", "DEFAULT", "DEVELOP"]))
@click.option("--sub_output_path", default="vocab", help=u"词典文件输出路径")
@click.option("--fields", default="", help=u"需要保留的字段,逗号分隔,默认全部保留")
@click.option("--data_dir", default="data", help=u"数据处理路径")
def generateVoca(mode, sub_output_path, fields, data_dir):
    u'''提取所有json文件的词典, 并将所有的review转换成为字典

    json文件的查找路径为当前路径之下的data/source

    词典输出路径为: data/preprocess/${sub_output_path}

    reviewText转换的文件输出在data/preprocess/transform/

    fields为一个逗号分隔的list,合法值如下:
        asin,helpful,overall,reviewText,
        reviewTime,reviewerID,reviewerName,
        summary,unixReviewTime
    选项不提供默认全部保留
    e.g --fields asin,reviewerID,overall,reviewText

    '''
    runConfig = config.configs[mode]("generateVoca_"+data_dir)
    runConfig.setConsoleLogLevel("DEBUG")
    runConfig.setSourcePath(data_dir)
    runConfig.checkValid()
    logger = runConfig.getLogger()
    logger.info("the command line is %s", " ".join(sys.argv))
    inputPath = runConfig.PREPROCESS_CONFIG["source_path"]
    outputPath = runConfig.PREPROCESS_CONFIG["output_path"]

    '''从source path下所有json文件中提取出词典'''
    vocas = []
    users = []
    items = []
    if not os.path.exists(os.path.join(outputPath, sub_output_path)):
        os.mkdir(os.path.join(outputPath, sub_output_path))

    for filename in os.listdir(inputPath):
        @recordTime
        def buildMapper(filename):
            voca, user, item = buildVoca(os.path.join(inputPath, filename))
            vocas.append(voca)
            users.append(user)
            items.append(item)
            outputDir = os.path.join(outputPath, sub_output_path)
            outputName = os.path.join(outputDir,
                                      filename.rsplit(".", 1)[0]+".pk")
            pkdump(voca, outputName)
            outputName = os.path.join(outputDir,
                                      filename.rsplit(".", 1)[0]+"_user.pk")
            pkdump(user, outputName)
            outputName = os.path.join(outputDir,
                                      filename.rsplit(".", 1)[0]+"_item.pk")
            pkdump(item, outputName)

        buildMapper(filename)

    voca = set()
    for v in vocas:
        voca.update(set(v.values()))
    voca = {i: word for i, word in enumerate(voca)}
    pkdump(voca, os.path.join(outputPath, "vocab", "allDomain.pk"))

    user = set()
    for v in users:
        user.update(set(v.values()))
    user = {i: u for i, u in enumerate(user)}
    pkdump(user, os.path.join(outputPath, "vocab", "allDomain_user.pk"))

    item = set()
    for v in items:
        item.update(set(v.values()))
    item = {i: it for i, it in enumerate(item)}
    pkdump(item, os.path.join(outputPath, "vocab", "allDomain_item.pk"))

    '''利用词典将原始文本得reviewText转换成文id'''
    transOutputPath = os.path.join(outputPath, "transform")
    if not os.path.exists(transOutputPath):
        os.mkdir(transOutputPath)

    voca = {word: i for i, word in voca.items()}
    user = {u: i for i, u in user.items()}
    item = {it: i for i, it in item.items()}
    mapper = {"vocab": voca, "user": user, "item": item}
    fields = None if fields == "" else fields.split(",")
    for filename in os.listdir(inputPath):
        @recordTime
        def trans(filename):
            transReview(os.path.join(inputPath, filename),
                        mapper, os.path.join(transOutputPath, filename),
                        fields)
        trans(filename)

    @recordTime
    def getColdUser():
        files = [os.path.join(inputPath, f) for f in os.listdir(inputPath)]
        SOURCE, TARGET = files[:2]
        ColdSU, ColdTU, OverLapU = generateColdUser(SOURCE, TARGET)
        logger.info("cold user count in %s is %d",
                    os.path.basename(SOURCE),
                    len(ColdSU))
        logger.info("cold user count in %s is %d",
                    os.path.basename(TARGET),
                    len(ColdTU))
        logger.info("overlap user count is %d", len(OverLapU))

        ColdOutputPath = os.path.join(outputPath, "cold")
        if not os.path.exists(ColdOutputPath):
            os.mkdir(ColdOutputPath)

        ColdSO = os.path.join(ColdOutputPath, os.path.basename(SOURCE))
        ColdTO = os.path.join(ColdOutputPath, os.path.basename(TARGET))
        pkdump(ColdSU, ColdSO.rsplit(".", 1)[0]+".pk")
        pkdump(ColdTU, ColdTO.rsplit(".", 1)[0]+".pk")
        pkdump(OverLapU, os.path.join(ColdOutputPath, "overlapUser.pk"))

    getColdUser()


@cli.command()
@click.option("--mode", type=click.Choice(
    ["DEBUG", "DEFAULT", "DEVELOP"]))
@click.option("--data_dir", default="data", help=u"数据处理路径")
def extractInfo(mode, data_dir):
    '''从转换后的用户数据提取信息'''
    runConfig = config.configs[mode]("extractInfo_"+data_dir)
    runConfig.setConsoleLogLevel("DEBUG")
    runConfig.setSourcePath(data_dir)
    runConfig.checkValid()
    logger = runConfig.getLogger()
    logger.info("the command line is %s", " ".join(sys.argv))



cli()
