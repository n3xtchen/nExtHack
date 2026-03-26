#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# Flink 版本参数 - 可在编译时覆盖
ARG FLINK_VERSION=2.2

FROM apache/flink:${FLINK_VERSION}-java21

SHELL ["/bin/bash", "-c"]

# 切换到 root 用户以进行安装
USER root

WORKDIR /opt/flink

# 激活 S3 文件系统插件 (Presto 实现 - 不需要 Hadoop 依赖)
RUN echo "-> 清理现有 S3 插件" && \
    rm -rf ./plugins/s3-fs-presto ./plugins/s3-fs-hadoop 2>/dev/null || true && \
    echo "✅ 清理完成" && \
    echo "-> 激活 Flink S3 文件系统插件 (Presto 实现)" && \
    mkdir -p ./plugins/s3-fs-presto && \
    cp ./opt/flink-s3-fs-presto-* ./plugins/s3-fs-presto/ && \
    ls -la ./opt/flink-s3-fs-presto-* && \
    echo "✅ S3 Presto 插件激活成功"

# 创建 flink-conf.yaml 配置 S3 端点
RUN echo "-> 创建 Flink 配置文件" && \
    cat > conf/flink-conf.yaml << 'EOF'
# Flink S3 Presto 配置
s3.path.style.access: true
s3.endpoint: http://localhost:9000
s3.access-key: n3xtchen
s3.secret-key: n3xtchen
s3.ssl.enabled: false
EOF

# 验证安装
RUN echo "-> 验证 S3 Presto 插件安装：" && \
    ls -lah ./plugins/s3-fs-presto/ && \
    echo "" && \
    echo "-> flink-conf.yaml 配置：" && \
    cat ./conf/flink-conf.yaml && \
    echo "" && \
    echo "✅ Docker 镜像构建完成（Flink + Presto S3）"

USER flink
