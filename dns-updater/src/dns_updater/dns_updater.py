#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""

import json
import tomllib
from typing import ParamSpecArgs
import sys

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkalidns.request.v20150109.AddDomainRecordRequest import AddDomainRecordRequest
from aliyunsdkalidns.request.v20150109.DescribeSubDomainRecordsRequest import DescribeSubDomainRecordsRequest
from aliyunsdkalidns.request.v20150109.UpdateDomainRecordRequest import UpdateDomainRecordRequest

class AliApi:

    def __init__(self, key, secret):
        self.access_key = key
        self.acces_secret = secret
        self._client = None

    @property
    def client(self) -> AcsClient:
        """初始化 client"""

        if self._client:
            return self._client
        self._client = AcsClient(self.access_key, self.acces_secret, 'cn-hangzhou')
        return self._client

    def add_subdomian_record(self, sub_domain, ip):
        """更新记录"""
        request = AddDomainRecordRequest()
        request.set_accept_format('json')
        request.set_Value(ip)
        request.set_Type("A")
        request.set_RR(www)
        request.set_DomainName(sub_domain)
        response = self.client.do_action_with_exception(request)
        print(str(response, encoding='utf-8')) 

    def describe_subdomain_record(self, sub_domain):
        """获取记录"""
        request = DescribeSubDomainRecordsRequest()
        request.set_accept_format('json')
        request.set_SubDomain(sub_domain)
        response = self.client.do_action_with_exception(request)
        print(str(response, encoding='utf-8')) 
        record = json.loads(response)["DomainRecords"]["Record"][0]
        return record["RR"], record["Value"], record["RecordId"]

    def update_subdomain_record(self, sub_domain, new_ip):
        """更新子域名IP"""

        rr, ip, record_id = self.describe_subdomain_record(sub_domain)
        if ip == new_ip:
            print("无需更新!")
        else:
            request = UpdateDomainRecordRequest()
            request.set_accept_format('json')
            request.set_RecordId(record_id)
            request.set_RR(rr)
            request.set_Type("A")
            request.set_Value(new_ip)
            response = self.client.do_action_with_exception(request)
            print(str(response, encoding='utf-8')) 

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
