# coding=utf8

from ..utils import *

import numpy as np
from collections import defaultdict


def mergeUserItem(filename):
    data = pkload(filename)
    user = defaultdict(list)
    item = defaultdict(list)
    for (u, i), vector in data.items():
        user[u].append(vector)
        item[i].append(vector)
    user = dict(user)
    item = dict(item)
    # 原始user的value是np.array的list, array的shape为[1, feature_size],
    # 通过concatenate之后变成[n, feature_size], 在经过mean 为[feature_size]
    # 既, user/item的向量表示用与之相关的向量均值表示
    user = {u: np.mean(np.concatenate(vs), axis=0) for u, vs in user.items()}
    item = {i: np.mean(np.concatenate(vs), axis=0) for i, vs in item.items()}

    return user, item
