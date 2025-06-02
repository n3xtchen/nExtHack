#! /bin/sh
#
# init_proj_struct.sh
# Copyright (C) 2025 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.
#


PROJ_NAME='demo'

mkdir -p "src/{$PROJ_NAME}" tests
touch LICENSE README.md "src/{$PROJ_NAME}/__init__.py"
