# -*- coding: utf-8 -*-

import pandas as pd

from .utils import readJson
from ..utils import pkload, pkdump


def generateColdUser(sourceDomain, targetDomain):
    '''生成冷用户

    source domain 冷用户定义如下:
        在target domain中有记录而在source domain中没有记录的用户
    target domain 冷用户定义类似

    Parameters
    ----------
    sourceDomain : str
        sourceDomain json数据文件路径

    targetDomain : str
        targetDomain json数据文件路径

    Returns
    -------
        coldUserSource : pd.DataFrame
    '''

    Source = list(readJson(sourceDomain))
    SourceUserItem = [(d["reviewerID"], d["asin"]) for d in Source]
    Traget = list(readJson(targetDomain))
    TargetUserItem = [(d["reviewerID"], d["asin"]) for d in Traget]
    uiS = pd.DataFrame(data=SourceUserItem, columns=["user", "sourceItem"])
    uiT = pd.DataFrame(data=TargetUserItem, columns=["user", "targetItem"])
    uS = uiS.groupby("user").count()
    uT = uiT.groupby("user").count()
    uBoth = pd.concat([uS, uT], axis=1).fillna(0)
    # coldUserSource 的user 来自于 Target domain, 在Source Domain中记录数量为0
    coldUserSource = uBoth.query("sourceItem == 0")
    # coldUserTarget 的user 来自于 Source Domain, 在Target Domain 中记录数量为0
    coldUserTarget = uBoth.query("targetItem == 0")
    overlapUser = uBoth.query("sourceItem != 0 and targetItem != 0")
    return list(coldUserTarget.index), list(coldUserSource.index), list(overlapUser.index)

