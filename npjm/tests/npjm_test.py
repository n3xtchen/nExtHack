#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""
测试 npjm
"""

import os
from typing import List

from npjm.projects import fetch_story_list

API_USER = os.environ.get("API_USER")
API_PASS = os.environ.get("API_PASS")
WOKRSPACE_ID = os.environ.get("WOKRSPACE_ID")

def test_fetch_story_list():
    """
    测试获取需求列表
    """
    stories = fetch_story_list(WOKRSPACE_ID, API_USER, API_PASS)
    assert isinstance(stories, List)
