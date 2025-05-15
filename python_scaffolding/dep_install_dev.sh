#! /bin/sh
#
# init.sh
# Copyright (C) 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.
#


PROJ_NAME='demo'

python -m venv "venv_{$PROJ_NAME}"
source "venv_{$PROJ_NAME}/bin/activate"

# pip 不支持使用 pyproject 安装
pip install pip-tools

python -m piptools compile -o requirements.txt pyproject.toml


# 编译开发环境包
python -m piptools compile --extra=dev -c requirements.txt  -o requirements-dev.txt pyproject.toml

# 编辑模式安装项目，便于调试
# pyproject 不支持 editable install 以来
# pytest 就可以不用设置 pythonpath
echo "-e ." >> requirements-dev.txt

# 安装开发环境包
python -m piptools sync requirements-dev.txt
# 和 pip install 区别是：严格同步 requirements-dev.txt，删除未列出的包

deactivate
