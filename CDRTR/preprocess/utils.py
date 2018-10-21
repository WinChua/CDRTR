# -*- coding: utf8 -*-

import json
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from
    https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def readJson(filename):
    '''读取filename中的json数据

    filename文件中的json数据为原始的Amazon数据格式。
    每行一个json, 每个json的keys为
    {
        "asin": "B00002243X",
        "helpful": [4, 4],
        "overall": 5.0,
        "reviewText": "some review",
        "reviewTime": "08 17, 2011",
        "reviewerID": "A3F73SC1LY51OO",
        "reviewerName": "Alan Montgomery",
        "summary": "some summary",
        "unixReviewTime": 1313539200
    }

    Parameters
    ----------
    filename : str
        json文件路径

    Yields
    ------
    json : dict
        生成文件中每一行的json的字典
    '''
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def buildVoca(filename):
    '''为原始json数据filename中的reviewText生成词典

    Parameters
    ---------
    filename : str
        json数据路径
    '''
    voca = set()
    item = set()
    user = set()
    for jsonData in readJson(filename):
        reviewLine = clean_str(jsonData["reviewText"])
        voca.update(set(i for i in reviewLine.split(" ")))
        item.add(jsonData["asin"])
        user.add(jsonData["reviewerID"])

    voca = {i: word for i, word in enumerate(voca)}
    user = {i: u for i, u in enumerate(user)}
    item = {i: it for i, it in enumerate(item)}
    return voca, user, item


def transReview(filename, mapper, outputfilename, fields=None):
    '''将原始文件中得reviewText转换成字典id

    将filename中每一条review json数据中得reviewText字段
    按照词典mapper, 进行word到词id得转换, 结果输出为outputfilename

    Parameters
    ----------
    filename : str
    mapper : dict
        词典 --> 形式为: {"vocab": word : id, "user": user : id, "item": item:id}
    outputfilename : str
    fields : list of str
        提取的字段
    '''
    with open(outputfilename, "w") as f:
        pass

    for lines in _transReview(filename, mapper, fields):
        with open(outputfilename, "a") as f:
            f.write("\n".join(lines))


def _transReview(filename, mapper, fields=None):
    '''辅助函数,对文件写入做缓冲'''
    lines = []
    for jsonData in readJson(filename):
        reviewLine = clean_str(jsonData["reviewText"])
        jsonData["reviewText"] = [mapper["vocab"][word] for word in reviewLine.split(" ")]
        jsonData["reviewerID"] = mapper["user"][jsonData["reviewerID"]]
        jsonData["asin"] = mapper["item"][jsonData["asin"]]
        if fields is not None:
            jsonData = {k: jsonData[k] for k in fields}
        lines.append(json.dumps(jsonData))
        if len(lines) == 10000:
            yield lines
            lines = []

    if lines != []:
        yield lines
