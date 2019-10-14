#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework3.py

    DESCRIPTION
        1. 创建一个2*2的数组，计算对角线上元素的和
        2. 创建一个长度为9的一维数据，数组元素0到8。将它重新变为3*3的二维数组
        3. 创建两个3*3的数组，分别将它们合并为3*6、6*3的数组后，拆分为3个数组（维数不限定）
        4. 说说numpy数组的优点

    MODIFIED  (MM/DD/YY)
        Na  11/06/2018

"""
__VERSION__ = "1.0.0.11062018"


# imports
import numpy as np
import random

# consts

# functions
"""1. 创建一个2*2的数组，计算对角线上元素的和
   2. 创建一个长度为9的一维数据，数组元素0到8。将它重新变为3*3的二维数组
   3. 创建两个3*3的数组，分别将它们合并为3*6、6*3的数组后，拆分为3个数组（维数不限定）
   4. 说说numpy数组的优点
        
"""
def main():
    # 1. 创建一个2*2的数组，计算对角线上元素的和
    array1 = np.array([[1,2], [3,4]])
    result = array1.diagonal().sum()
    print(u'{0} 该数组对角线上元素的和 = {1}.'.format(array1, result))

    # 2. 创建一个长度为9的一维数据，数组元素0到8。将它重新变为3*3的二维数组
    array2 = np.arange(9)
    print(u'\n一维数组: {0}'.format(array2))
    array2 = array2.reshape(3, 3)
    print(u'二维数组:\n{0}'.format(array2))

    # 3. 创建两个3*3的数组，分别将它们合并为3*6、6*3的数组后，拆分为3个数组（维数不限定）
    array3 = np.array([random.randrange(1, 100) for _ in xrange(9)]).reshape(3, 3)
    array4 = np.array([random.random() for _ in xrange(9)]).reshape(3, 3)
    print(u'\n两个3*3的数组：\n{0}\n{1}'.format(array3, array4))

    array_hstack = np.hstack((array3, array4))
    print(u'合并为3*6的数组：\n{0}'.format(array_hstack))
    array36_hsplit = np.hsplit(array_hstack, 3)
    print(u'合并为3*6的数组后，沿0轴拆分为3个数组：\n{0}'.format(array36_hsplit))
    array36_vsplit = np.vsplit(array_hstack, 3)
    print(u'合并为3*6的数组后，沿1轴拆分为3个数组：\n{0}'.format(array36_vsplit))

    array_vstack = np.vstack((array3, array4))
    print(u'\n合并为6*3的数组：\n{0}'.format(array_vstack))
    array63_hsplit = np.hsplit(array_vstack, 3)
    print(u'合并为6*3的数组后，沿0轴拆分为3个数组：\n{0}'.format(array63_hsplit))
    array63_vsplit = np.vsplit(array_vstack, 3)
    print(u'合并为6*3的数组后，沿1轴拆分为3个数组：\n{0}'.format(array63_vsplit))

    # 4. 说说numpy数组的优点
    np_advantages = u'''
    1.灵活高效的N维数组，例如灵活的调整维数，因为代码是C实现的、运算很高效，等等
    2.numpy提供了很多对数组进行元素级操作和矩阵操作的函数
    3.如果dtype相同，可以很简便的算出数组需要的存储大小
    4.可以用来生成pandas库中的DataFrame
    等等。
    '''
    print(u'\nnumpy数组的优点:\n{0}'.format(np_advantages))

# classes

# main entry
if __name__ == "__main__":
    main()
    