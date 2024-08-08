#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""
Ali API
"""

import configparser
from pathlib import Path
import json

from aliyunsdkalidns.request.v20150109.AddDomainRecordRequest import (
    AddDomainRecordRequest, )
from aliyunsdkalidns.request.v20150109.DescribeSubDomainRecordsRequest import (
    DescribeSubDomainRecordsRequest, )
from aliyunsdkalidns.request.v20150109.UpdateDomainRecordRequest import (
    UpdateDomainRecordRequest, )
from aliyunsdkcore.client import AcsClient

DEFAULT_CREDENTIALS_FILE = str(Path.home()) + "/.alibabacloud/credentials.ini"

class AliApi:
    """Ali API"""

    def __init__(self, credentials_file: str=""):
        """初始化"""

        self.credentials_file:str = credentials_file or DEFAULT_CREDENTIALS_FILE

        self._config = None
        self._client = None

    @property
    def config(self):
        """读取密钥"""
        if self._config is None:
            config = configparser.ConfigParser(allow_no_value=True)
            config.read(self.credentials_file)
            self._config = config
        return self._config

    @property
    def client(self) -> AcsClient:
        """初始化 client"""

        if self._client:
            return self._client
        self._client = AcsClient(self.config["default"]["access_key_id"],
                                 self.config["default"]["access_key_secret"],
                                 'cn-hangzhou')
        return self._client

    def add_subdomian_record(self, sub_domain, ip):
        """更新记录"""
        request = AddDomainRecordRequest()
        request.set_accept_format('json')
        request.set_Value(ip)
        request.set_Type("A")
        request.set_RR("www")
        request.set_DomainName(sub_domain)
        response = self.client.do_action_with_exception(request)
        return response

    def describe_subdomain_record(self, sub_domain):
        """获取记录"""
        request = DescribeSubDomainRecordsRequest()
        request.set_accept_format('json')
        request.set_SubDomain(sub_domain)
        response = self.client.do_action_with_exception(request)
        record = json.loads(str(response))["DomainRecords"]["Record"][0]
        return record["RR"], record["Value"], record["RecordId"]

    def update_subdomain_record(self, sub_domain, new_ip):
        """更新子域名IP"""

        rr, ip, record_id = self.describe_subdomain_record(sub_domain)
        if ip == new_ip:
            print("无需更新!")
            return None

        request = UpdateDomainRecordRequest()
        request.set_accept_format('json')
        request.set_RecordId(record_id)
        request.set_RR(rr)
        request.set_Type("A")
        request.set_Value(new_ip)
        response = self.client.do_action_with_exception(request)
        print("更新成功！")
        return response
