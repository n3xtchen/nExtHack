#! /bin/sh
#
# dep_install_prod.sh
# Copyright (C) 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.
#


PROJ_NAME='demo'

python -m venv "venv_{$PROJ_NAME}"
source "venv_{$PROJ_NAME}/bin/activate"

# 保证线上包版本的稳定性
python -m pip freeze > constraints.txt

# pip 不支持使用 pyproject 安装
pip install pip-tools

python -m piptools compile -c constraints.txt -o requirements.txt pyproject.toml

# 安装生产环境包
python -m piptools sync requirements.txt

deactivate

