# coding=utf8

import os
import pickle
import time
import functools
import logging


def padding(sentc, size, pad):
    '''对sentence进行padding操作

    Parameters
    ----------
    sentc : list
    size  : int
    pad   : type of list[0]

    Returns
    -------
    list : padding result
    '''
    if len(sentc) < size:
        return sentc + [pad] * (size - len(sentc))
    else:
        return sentc[:size]


def pkdump(obj, filename):
    '''pickle 包装函数'''

    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pkload(filename):
    '''pickle 包装函数'''
    with open(filename, "rb") as f:
        return pickle.load(f)


def recordTime(func):
    logger = logging.getLogger(__name__)

    @functools.wraps(func)
    def innerWrapper(*args, **kwargs):
        logger.debug("calling  func: %s, args: %s, kwargs: %s",
                     func.func_name,
                     " ".join(args),
                     " ".join("{k}={v}".format(k=k, v=v)
                              for k, v in kwargs.items())
                     )
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("call end func: %s, it cost %f s",
                     func.func_name, end - start)
        return result

    return innerWrapper
