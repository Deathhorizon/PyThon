#!/usr/bin/env python 
# -*- coding:utf-8 -*-
def k_equal(a_list, k):
    """
    K等分点
    :param a_list: n边形坐标列表
    :param k: 分成k段
    :return: 等分点的坐标
    """
    n = len(a_list)
    lines_len = 0  # 定义线段长度
    line_direction = []  # 1代表垂直上，2代表水平右，-1代表垂直下，-2代表水平左
    line_len_list = []  # 记录每段线段的长度

    for i in range(n - 1):
        # 如果x坐标相等，就是垂直线段，y坐标差就是线段长度
        if a_list[i][0] == a_list[i + 1][0]:
            line_len = abs(a_list[i + 1][1] - a_list[i][1])
            lines_len += line_len
            line_len_list.append(line_len)
            if a_list[i + 1][1] - a_list[i][1] > 0:
                line_direction.append(1)
            else:
                line_direction.append(-1)
        # 如果y坐标相等，就是水平线段，x坐标差就是线段长度
        if a_list[i][1] == a_list[i + 1][1]:
            line_len = abs(a_list[i + 1][0] - a_list[i][0])
            lines_len += line_len
            line_len_list.append(line_len)
            if a_list[i + 1][0] - a_list[i][0] > 0:
                line_direction.append(2)
            else:
                line_direction.append(-2)

    # 加上终点到起点的线段
    if a_list[0][0] == a_list[n - 1][0]:
        line_len = abs(a_list[0][1] - a_list[n - 1][1])
        lines_len += line_len
        line_len_list.append(line_len)
        if a_list[0][1] - a_list[n - 1][1] > 0:
            line_direction.append(1)
        else:
            line_direction.append(-1)
    if a_list[0][1] == a_list[n - 1][1]:
        line_len = abs(a_list[0][0] - a_list[n - 1][0])
        lines_len += line_len
        line_len_list.append(line_len)
        if a_list[0][0] - a_list[n - 1][0] > 0:
            line_direction.append(2)
        else:
            line_direction.append(-2)

    equal_line = lines_len / k  # 等分线段长度
    tmp_equal_line = equal_line  # 存储等分线段长度
    m = len(line_direction)
    k_list = []  # 存储k等分点坐标

    for j in range(m):
        # 垂直上线段
        if line_direction[j] == 1:
            # 当前线段上有多个等分点的情况
            if equal_line <= line_len_list[j]:
                base = 0  # 记录当前k坐标起点
                while equal_line <= line_len_list[j]:
                    k_list.append((float(a_list[j][0]), a_list[j][1] + (equal_line + base)))
                    base += equal_line
                    line_len_list[j] -= equal_line
                    equal_line = tmp_equal_line
                equal_line = tmp_equal_line - line_len_list[j]
            else:
                equal_line -= line_len_list[j]
        # 垂直下线段
        elif line_direction[j] == -1:
            # 当前线段上有多个等分点的情况
            if equal_line <= line_len_list[j]:
                base = 0  # 记录当前k坐标起点
                while equal_line <= line_len_list[j]:
                    k_list.append((float(a_list[j][0]), a_list[j][1] - (equal_line + base)))
                    base += equal_line
                    line_len_list[j] -= equal_line
                    equal_line = tmp_equal_line
                equal_line = tmp_equal_line - line_len_list[j]
            else:
                equal_line -= line_len_list[j]
        # 水平右线段
        elif line_direction[j] == 2:
            # 当前线段上有多个等分点的情况
            if equal_line <= line_len_list[j]:
                base = 0  # 记录当前k坐标起点
                while equal_line <= line_len_list[j]:
                    k_list.append((a_list[j][0] + (equal_line + base), float(a_list[j][1])))
                    base += equal_line
                    line_len_list[j] -= equal_line
                    equal_line = tmp_equal_line
                equal_line = tmp_equal_line - line_len_list[j]
            else:
                equal_line -= line_len_list[j]
        # 水平左线段
        else:
            # 当前线段上有多个等分点的情况
            if equal_line <= line_len_list[j]:
                base = 0  # 记录当前k坐标起点
                while equal_line <= line_len_list[j]:
                    k_list.append((a_list[j][0] - (equal_line + base), float(a_list[j][1])))
                    base += equal_line
                    line_len_list[j] -= equal_line
                    equal_line = tmp_equal_line
                equal_line = tmp_equal_line - line_len_list[j]
            else:
                equal_line -= line_len_list[j]

    print(k_list)


alist = [(1, 2), (1, 3), (5, 3), (5, 2)]
k_equal(alist, 5)
