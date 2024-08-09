#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""

from dns_updater.dns_updater import get_outer_ip,update_dns_record
from dns_updater.AliApi import AliApi

SUBDOMAIN = "home.n3xt.space"

def test_credential():
    """测试阿里Key"""

    ali = AliApi()
    print(ali.config.sections())

def test_get_dns_record():
    """测试查询dns记录"""

    ali = AliApi()
    ali.describe_subdomain_record(SUBDOMAIN)

def test_update_dns_record_with_ip():
    """测试更新dns记录"""
    ali = AliApi()
    ali.update_subdomain_record(SUBDOMAIN, "120.36.84.161")

def test_ip():
    """测试外网IP"""
    print(get_outer_ip())

def test_update_dns_record():
    """更新 dns 记录"""
    update_dns_record()

