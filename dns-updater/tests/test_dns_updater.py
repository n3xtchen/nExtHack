#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""

import tomllib

from dns_updater.AliApi import AliApi

def test_answer():
    """test"""
    with open('config.toml', 'rb') as f:
        config = tomllib.load(f)
        print(config)

        print(AliApi)
        # ali = dns_updater.AliApi(config["access_key"], config["access_secret"])
        # ali.describe_subdomain_record(config["sub_domain"])
