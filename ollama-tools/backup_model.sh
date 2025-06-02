#! /bin/bash
#
# backup_model.sh
# Copyright (C) 2024 n3xtchen <echenwen@gmail.com>
#
# Distributed under terms of the GPL-2.0 license.
#

BLOBS="models/blobs"
MANIFEST="models/manifests/"

OLLAMA_HOME="${HOME}/.ollama"

TARGET_DIR=$1
MODEL_NAME=$2
MODEL_TAG=$3
DEFAULT_REGISTRY=${4:-"registry.ollama.ai"}

MANIFEST="models/manifests/${DEFAULT_REGISTRY}/library/${MODEL_NAME}/${MODEL_TAG}"

SRC_MANIFEST="${OLLAMA_HOME}/${MANIFEST}"
SRC_MODEL_BLOBS="${OLLAMA_HOME}/models/blobs"

TARGET_MANIFEST="${TARGET_DIR}/${MANIFEST}"
TARGET_MANIFEST_DIR=${TARGET_MANIFEST%/*} 
TARGET_MODEL_BLOBS="${TARGET_DIR}/models/blobs"

if [ -f "${SRC_MANIFEST}" ]; then
  if [ -d "$TARGET_MANIFEST_DIR" ]; then
    echo "$TARGET_MANIFEST_DIR 已存在"
  else
    echo "$TARGET_MANIFEST_DIR 不存在, 开始创建..."
    mkdir -p $TARGET_MANIFEST_DIR
    echo "创建成功"
  fi

  if [ -f "$TARGET_MANIFEST" ]; then
    echo "MANIFEST 文件存在！检查一致性.."

    if diff -q $SRC_MANIFEST $TARGET_MANIFEST > /dev/null; then
      echo "MANIFEST 文件相同"
    else
      echo "MANIFEST 文件不同！开始复制..."
      cp $SRC_MANIFEST $TARGET_MANIFEST_DIR
    fi
  else
    echo "开始拷贝 MANIFEST..."
    cp $SRC_MANIFEST $TARGET_MANIFEST_DIR
  fi
else
  echo "模型不存在"
  exit 1
fi

grep -oE "sha256:[a-fA-F0-9]{64}" $SRC_MANIFEST  | sed 's/:/-/' | while read line; do

  SRC_MODEL_BLOB="${SRC_MODEL_BLOBS}/${line}"
  TARGET_MODEL_BLOB="${TARGET_MODEL_BLOBS}/${line}"

  if [ -f "${TARGET_MODEL_BLOB}" ]; then
    if cmp -s $SRC_MODEL_BLOB $TARGET_MODEL_BLOB > /dev/null; then
        echo "${line} 已经存在！"
    else  
        echo "${line} 文件不同！开始复制..."
        cp $SRC_MODEL_BLOB $TARGET_MODEL_BLOBS
    fi
  else
    echo "${line} 文件不存在！开始复制..."
    cp $SRC_MODEL_BLOB $TARGET_MODEL_BLOBS
  fi
done