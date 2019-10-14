#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    NAME
        housework2.py

    DESCRIPTION
        1. 编写函数，要求输入x与y，返回x和y的平方差
        2. 计算1到100的平方的和
        3. 编写函数，若输入为小于100的数，返回TRUE，大于100的数，返回FALSE
        4. 某个公司采用公用电话传递数据，数据是四位的整数，在传递过程中是加密的，加密规则如下：
        每位数字都加上5,然后用和除以10的余数代替该数字，再将第一位和第四位交换，第二位和第三位交换。
        编写加密的函数与解密的函数。

    MODIFIED  (MM/DD/YY)
        Na  11/06/2018

"""
__VERSION__ = "1.0.0.11062018"


# imports
import sys

# consts

# functions
"""1. 编写函数，要求输入x与y，返回x和y的平方差"""
def square_diff():
    # get inputs
    x = y = x_str = y_str = None
    while x is None:
        x_str = raw_input(u'计算x和y的平方差，请输入x：'.encode(sys.stdin.encoding))
        try:
            x = int(x_str)
        except ValueError as e:
            try:
                x = float(x_str)
            except ValueError as e:
                print(u'x类型不对。请输入整数或者小数。')
    while y is None:
        y_str = raw_input(u'计算x和y的平方差，请输入y:'.encode(sys.stdin.encoding))
        try:
            y = int(y_str)
        except ValueError as e:
            try:
                y = float(y_str)
            except ValueError as e:
                print(u'y类型不对。请输入整数或者小数。')
    # calculate square difference
    result = x**2 - y**2
    print(u'[INFO] x和y的平方差: {0}^2 - {1}^2 = {2}'.format(x, y, result))
    return result

"""2. 计算1到100的平方的和"""
def sum(N=100):
    if N >= 1:
        result = 0
        for i in xrange(1, N+1):
            result += i**2
        print(u'1到{0}的平方的和为{1}.'.format(N, result))
        return result
    else:
        # print(u'计算1到N的平方的和，请输入大于1的整数为N。')
        pass

"""3. 编写函数，若输入为小于100的数，返回TRUE，大于100的数，返回FALSE"""
def check_input():
    input_num = input_num_str = None
    while input_num is None:
        input_num_str = raw_input(u'请输入一个数字：'.encode(sys.stdin.encoding))
        try:
            input_num = int(input_num_str)
        except ValueError as e:
            try:
                input_num = float(input_num_str)
            except ValueError as e:
                print(u'输入的类型不是数字,请输入整数或者小数。')
    # check if > 100
    if input_num < 100:
        print(u'输入的数字是小于100的数：{0}'.format(input_num))
        return True
    else:
        print(u'输入的数字是大于等于100的数：{0}'.format(input_num))
        return False

"""4. 某个公司采用公用电话传递数据，数据是四位的整数，在传递过程中是加密的，加密规则如下：
    每位数字都加上5,然后用和除以10的余数代替该数字，再将第一位和第四位交换，第二位和第三位交换。
    编写加密的函数与解密的函数。

"""
def encrypt_data(plain_data=None):
    if plain_data and isinstance(plain_data, int) and 1000 <= plain_data <= 9999:
        n1, n2, n3, n4 = map(lambda c: (int(c)+5)%10, str(plain_data))
        result = n4 * 1000 + n3 * 100 + n2 * 10 + n1
        print(u'整数{0}加密以后为{1}'.format(plain_data, result))
        return result
    else:
        print(u'请使用四位整数作为待加密数据。')

def decrypt_data(encryted_data=None):
    if encryted_data and isinstance(encryted_data, int) and 1000 <= encryted_data <= 9999:
        n4, n3, n2, n1 = map(lambda c: (int(c)+5)%10, str(encryted_data))
        result = n1 * 1000 + n2 * 100 + n3 * 10 + n4
        print(u'整数{0}解密以后为{1}'.format(encryted_data, result))
        return result
    else:
        print(u'请使用四位整数作为待解密数据。')

# classes

# main entry
if __name__ == "__main__":
    square_diff()
    sum(N=10)
    # check_input()
    decrypt_data(encrypt_data(1234))
    decrypt_data(encrypt_data(5678))



    