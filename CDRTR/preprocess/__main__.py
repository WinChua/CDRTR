# -*- coding: utf8 -*-

import os
import sys
import click

from itertools import chain
from operator import itemgetter
from glob import glob
from .utils import buildVoca, transReview
from .utils import readJson
from ..utils import pkdump, recordTime
from .. import config
from .cold import generateColdUser
from .sentiOutputMergeUserItem import mergeUserItem


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
            fnPrefix = filename.rsplit(".", 1)[0]
            for suffix, data in zip((".pk", "_user.pk", "_item.pk"),
                                    (voca, user, item)):
                outputName = os.path.join(outputDir, fnPrefix + suffix)
                pkdump(data, outputName)

        buildMapper(filename)

    mappers = []
    for suffix, data in zip((".pk", "_user.pk", "_item.pk"),
                            (vocas, users, items)):
        dumpData = set()
        for d in data:
            dumpData.update(set(d.values()))
        dumpData = {i: content for i, content in enumerate(dumpData)}
        pkdump(dumpData, os.path.join(outputPath, "vocab", "allDomain"+suffix))
        mappers.append(dumpData)

    '''利用词典将原始文本得reviewText转换成文id'''
    transOutputPath = os.path.join(outputPath, "transform")
    if not os.path.exists(transOutputPath):
        os.mkdir(transOutputPath)

    voca, user, item = mappers
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
    u'''从转换后的用户数据提取信息'''
    runConfig = config.configs[mode]("extractInfo_"+data_dir)
    runConfig.setConsoleLogLevel("DEBUG")
    runConfig.setSourcePath(data_dir)
    runConfig.checkValid()
    logger = runConfig.getLogger()
    logger.info("the command line is %s", " ".join(sys.argv))


@cli.command()
@click.option("--mode", type=click.Choice(["DEBUG", "DEFAULT", "DEVELOP"]))
@click.option("--data_dir", default="data", help=u"数据路径")
@click.option("--domain", default="*", help=u"待处理领域, 逗号分隔")
@click.option("--output_dir", default="uirepresent", help=u"输出路径")
def mergeUI(mode, data_dir, domain, output_dir):
    runConfig = config.configs[mode]("mergeUI_"+data_dir)
    runConfig.setConsoleLogLevel("DEBUG")
    logger = runConfig.getLogger()
    pattern = "%s/preprocess/sentiRecOutput/%s"

    patterns = [glob(pattern % (data_dir, d)) for d in domain.split(",")]

    out_dir = "%s/preprocess/%s" % (data_dir, output_dir)
    for fn in chain(*patterns):
        logger.debug("Dealing with %s", fn)
        user, item = mergeUserItem(fn)
        bn = os.path.basename(fn)
        pkdump(user, os.path.join(out_dir, "user_"+bn))
        pkdump(item, os.path.join(out_dir, "item_"+bn))
        logger.debug("Dealing end  %s", fn)

    rating = "%s/preprocess/transform/*.json"
    key_getter = itemgetter("reviewerID", "asin")
    value_getter = itemgetter("overall")
    for fn in glob(rating % data_dir):
        data = {key_getter(d): value_getter(d) for d in readJson(fn)}
        out_name = os.path.join(out_dir,
                                os.path.basename(os.path.splitext(fn)[0])+".pk")
        out_name = out_name.replace("reviews", "rating")
        pkdump(data, out_name)


cli()
