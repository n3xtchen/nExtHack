#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.
"""

"""

import argparse
import urllib.request

from dns_updater.ali_api import AliApi

def get_outer_ip():
    """获取外网IP"""

    with urllib.request.urlopen('https://ifconfig.me/ip') as response:
        return response.read().decode("utf-8")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('subdomain')
    parser.add_argument('--ip', default=None)
    args = parser.parse_args()
    ip = args.ip or get_outer_ip()
    ali = AliApi()
    ali.update_subdomain_record(args.subdomain, ip)
