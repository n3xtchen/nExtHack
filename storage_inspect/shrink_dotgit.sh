#! /bin/sh
#
# shrink_dotgit.sh
# Copyright (C) 2023 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.
#

du -h --max-depth=1 .git

# time git lfs prune
time git gc
time git prune
time git repack -a -d --depth=250 --window=250
time git gc --aggressive --prune

du -h --max-depth=1 .git
