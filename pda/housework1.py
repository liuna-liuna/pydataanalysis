#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        lesson1.py

    DESCRIPTION
         （1）创建0~10的列表
        （2）创建0~10的元组
        （3）打印列表与元组的内容

    MODIFIED  (MM/DD/YY)
        Na  10/23/18 - created

"""
__VERSION__ = "1.0.0.102318"


# imports

# consts

# functions

# classes
class PDA_Lesson1(object):
    def __init__(self):
        pass

    def run(self, *args, **kwargs):
        self._print_demo1()

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    def _print_demo1(self):
        list1 = range(11)
        print 'list 0~10:\n\t', list1, ', ', type(list1)
        tuple1 = tuple(i for i in range(11))
        print 'tuple 0~10:\n\t', tuple1, ', ', type(tuple1)

# main entry
if __name__ == "__main__":
    PDA_Lesson1()()
    