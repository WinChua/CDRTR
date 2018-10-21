#! -*- coding: utf8 -*-

def autofunc(hello, world):
    '''一行概括性描述
    详细描述


   :param int hello: hello参数, 整数类型. 用于返回宇宙的终极答案
   :param str world: world参数, 字符串类型. 用于存储world值

   :returns:
      * **hello** (*int*) -- hello
      * **world** (*str*) -- world
    '''
    return 42

def googleStyleDoc(arg1, arg2):
    '''

    Args:
        arg1: int
            描述一下这个参数
        arg2: int
            描述一下这个参数

    Returns:
        hello: str
        world: str

    '''
    return "hello", "world"

def numpyStyleDoc(arg1, arg2):
    '''利用sphinxcontrib.napoleon支持numpy style的docstrings

    hello world

    Parameters
    ----------
    arg1 : int
       描述一下这个参数1
    arg2 : str
       描述一下这个参数2

    Returns
    -------
    hello : int
       描述一下这个返回值
    world : str
       描述一下这个返回值
    '''

class HW:
    '''文档demo类
    闻声好

    演示一下
    '''
    def hello(self, world):
        '''问好

        Args:
            world: str
                核来

        Returns:
            int : 总是42

        '''
        return 42