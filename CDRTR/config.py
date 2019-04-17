# coding=utf-8
import os
import logging
import logging.config

import tensorflow as tf

dirname = os.getcwd()


class Config:
    '''保存默认配置的基类, 配置选项包括:

    * 预处理配置
    * tensorflow 配置
    * 日志配置
    '''

    _logger = None

    MODE = "DEFAULT"
    DATADIR = "{data}"
    PREPROCESS_CONFIG = {
        "source_path": os.path.join(dirname, DATADIR, "source"),
        "output_path": os.path.join(dirname, DATADIR, "preprocess")
    }

    @classmethod
    def setSourcePath(cls, path):
        for k in cls.PREPROCESS_CONFIG:
            cls.PREPROCESS_CONFIG[k] = cls.PREPROCESS_CONFIG[k].format(data=path)

    TF_CONFIG = {
        "per_process_gpu_memory_fraction": 0.5,
        "allow_growth": True
    }

    @classmethod
    def setGpuOption(cls):
        config = tf.ConfigProto()
        logger = cls.getLogger()
        for k, v in cls.TF_CONFIG.items():
            setattr(config.gpu_options, k, v)
            logger.info("GPU Option Setting: key--%s, value--%s", k, str(v))
        return config

    LOG_CONFIG = {
        "version": 1,  # version字段是必须的，当前合法值只有1
        "formatters": {
            "file_line": {
                "class": "logging.Formatter",
                "format": "[%(levelname)1.1s %(asctime)s %(name)s:%(lineno)d] %(message)s"
            }
        },
        "filters": {},
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": os.path.join("log", "default.log"),
                "mode": "w",
                "formatter": "file_line",
            },
            "console": {
                "class": "logging.StreamHandler",
                "level": "ERROR",
                "formatter": "file_line",
            }
        },
        "loggers": {},
        "root": {
            "level": "DEBUG",
            "handlers": ["file", "console"]
        }
    }

    @classmethod
    def getGPUConfig(cls):
        config = tf.ConfigProto()
        for key, value in cls.TF_CONFIG.items():
            setattr(config.gpu_options, key, value)
        return config

    @classmethod
    def checkValid(cls):
        if not os.path.isdir(cls.PREPROCESS_CONFIG["source_path"]):
            raise Exception("the source path of preprocess: %s is not"
                            " a directory." % cls.PREPROCESS_CONFIG["source_path"])
        if not os.path.isdir(cls.PREPROCESS_CONFIG["output_path"]):
            os.mkdir(cls.PREPROCESS_CONFIG["output_path"])

        return True

    @classmethod
    def setConsoleLogLevel(cls, level):
        cls.LOG_CONFIG["handlers"]["console"]["level"] = level

    @classmethod
    def setLogFile(cls, filename):
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        cls.LOG_CONFIG["handlers"]["file"]["filename"] = filename

    @classmethod
    def setLogLevel(cls, level):
        cls.LOG_CONFIG["root"]["level"] = level

    @classmethod
    def getLogger(cls):
        if cls._logger is None:
            logging.config.dictConfig(cls.LOG_CONFIG)
            cls._logger = logging.getLogger()
        return cls._logger


class DebugConfig(Config):
    '''调试配置'''
    MODE = "DEBUG"

    @classmethod
    def setGetConfig(cls, filename):
        dir = os.path.dirname(filename)
        fn = os.path.basename(filename)
        dir = os.path.join(dirname, "log", dir)
        cls.setLogFile(os.path.join(dir, "debug_%s.log" % fn))
        return cls


class DevelopConfig(Config):
    '''实验配置'''
    MODE = "DEVELOP"

    @classmethod
    def setGetConfig(cls, filename):
        dir = os.path.dirname(filename)
        fn = os.path.basename(filename)
        dir = os.path.join(dirname, "log", dir)
        cls.setLogFile(os.path.join(dir, "develop_%s.log" % fn))
        cls.setLogLevel("INFO")
        return cls


configs = {
    DebugConfig.MODE: lambda filename: DebugConfig.setGetConfig(filename),
    DevelopConfig.MODE: lambda filename: DevelopConfig.setGetConfig(filename),
    Config.MODE: lambda: Config
}

__all__ = ["configs"]
