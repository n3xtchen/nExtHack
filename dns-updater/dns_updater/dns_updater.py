#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.
"""

"""

import sys
import tomllib
from typing import ParamSpecArgs

from dns_updater.AliApi import AliApi

if __name__ == "__main__":

    # 读取 toml 文件
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
        print(config)

        ali = AliApi(config["access_key"], config["access_secret"])
        ali.describe_subdomain_record(config["sub_domain"])
        if len(sys.argv) == 2:
            ali.update_subdomain_record(config["sub_domain"], sys.argv[1])
            ali.describe_subdomain_record(config["sub_domain"])
        else:
            print("参数异常", sys.argv)
