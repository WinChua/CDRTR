# coding=utf8

import pickle
import time
import functools
import logging


def pkdump(obj, filename):
    '''pickle 包装函数'''
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
