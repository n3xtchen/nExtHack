#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.

"""
文件磁盘占用分析
"""

import os
import typer

MIN_FILE_SIZE = 1024 * 1024 * 10  # 设置最小文件大小为10MB，您可以根据需要调整大小

def scan_large_files(folder_path: str, min_file_size: int=MIN_FILE_SIZE):
    """
    扫描大文件
    """
    large_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.islink(file_path):
                file_size = os.path.getsize(file_path)
                if file_size > min_file_size:
                    large_files.append((file_path, file_size))
    if large_files:
        print("大文件列表：")
        for file_path, file_size in large_files:
            print(f"{file_path} ({file_size} bytes)")
    else:
        print("没有找到大文件。")

if __name__ == "__main__":
    typer.run(scan_large_files)
