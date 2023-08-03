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
from typing import List
import typer

MIN_FILE_SIZE = 1024 * 1024 * 10  # 设置最小文件大小为10MB，您可以根据需要调整大小

def sizeof_fmt(num, suffix="B"):
    """
    可视化字节数
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

def get_dir_size(folder_path: str):
    """
    获取目录磁盘占用
    """
    folder_size: int = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.islink(file_path):
                folder_size += os.path.getsize(file_path)
    return folder_size

def scan_large_files(folder_path: str, dir_pattern: List[str]=[], min_file_size: int=MIN_FILE_SIZE):
    """
    扫描大文件
    """
    large_files = []
    for root, folders, files in os.walk(folder_path):
        if dir_pattern != "":
            for folder in folders:
                folder_path = os.path.join(root, folder)
                if folder in dir_pattern and  not os.path.islink(folder_path):
                    folder_size = get_dir_size(folder_path)
                    if folder_size > min_file_size:
                        large_files.append((folder_path, folder_size))
        else:
            for file in files:
                file_path = os.path.join(root, file)
                if not os.path.islink(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > min_file_size:
                        large_files.append((file_path, file_size))
    if large_files:
        print("大文件列表：")
        for file_path, file_size in large_files:
            print(f"{file_path}\t({sizeof_fmt(file_size)})")
    else:
        print("没有找到大文件。")

if __name__ == "__main__":
    typer.run(scan_large_files)
