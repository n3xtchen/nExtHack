#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2023 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""

"""

import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData, Table
from sqlalchemy.orm import sessionmaker

ENGINE = create_engine(
    "mysql+pymysql://root:@127.0.0.1:3306/test?charset=utf8",
)

def table_info(table_name):
    meta_data = MetaData()
    meta_data.reflect(ENGINE)
    table = meta_data.tables[table_name]
    field_summary = "\n  ".join([
        f.name + " " + str(f.type) + (f" COMMENT \"{f.comment}\"" if(f.comment) else "")
    for f in table.columns])

    sample_data = ""
    with ENGINE.begin() as connection:
        result = connection.execute(table.select()).fetchall()
        if (len(result)>0):
            data = pd.DataFrame(result)
            data = data.to_string(index=False)
            sample_data = f"""{table.name} 的样本数据如下：
{data}"""

    return f"""{table.name} 的表结构如下：
  {field_summary}    
{sample_data}"""

def build_prompt(question, related_tables, opt_tips="", check_tips=""):
    question = question.strip()
    tables_summary = "\n\n".join(table_info(t) for t in related_tables)

    set_role = """现在你是一个资深的分析师，有着丰富分析经验和SQL能力，请你帮个忙"""

    prompt = f"""{set_role}
{opt_tips}
{tables_summary}

根据以上提供的数据结构和数据，编写一个具体和准确的 {ENGINE.dialect.name} 的查询语句来回答如下分析问题：

"{question}"
{check_tips}
并使用你的逻辑来注释它。 """
    
    return prompt
