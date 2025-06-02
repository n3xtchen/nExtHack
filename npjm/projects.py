#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.
"""
项目
"""

import requests

def fetch_story_list(workspace_id, user, passwd):
    """
    获取需求列表
    """

    req = requests.get(
        f"https://api.tapd.cn/stories?workspace_id={workspace_id}",
        auth=(user, passwd),
        timeout=200)

    stories = req.json()["data"] or []

    for row in stories:
        print(row["Story"]["name"])

    return req.json()["data"]
