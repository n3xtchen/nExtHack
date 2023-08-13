#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.
"""
依赖管理
"""

import sys


def main():
    """
    解析依赖
    """
    # 从标准输入中读取数据
    lines = sys.stdin.readlines()

    dep_cnt = {}

    # 处理数据
    for line in lines:
        to_from = line.split(":")
        dep_cnt[to_from[0]] = dep_cnt.get(to_from[0], 0)
        for dep in to_from[1].split():
            dep_cnt[dep] = dep_cnt.get(dep, 0) + 1

    for dep, cnt in dep_cnt.items():
        if cnt == 0:
            print(dep)


if __name__ == "__main__":
    main()
