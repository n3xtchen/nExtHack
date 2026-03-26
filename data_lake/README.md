# Flink S3 环境 - 项目 README

## 📌 项目概述

本项目为 Apache Flink 集成 S3（通过 RustFS）的开发环境配置。包含自动化部署脚本、Docker 配置和详细的部署指南。

**核心特性**：
- ✅ 基于 Apache Iceberg Flink Quickstart 官方项目
- ✅ 集成 RustFS S3 兼容存储
- ✅ 一键启动和停止脚本
- ✅ 完整的部署和测试文档
- ✅ 支持环境变量注入（安全的凭据管理）

---

## 🚀 快速开始

### 前置要求

- Docker 和 Docker Compose 已安装
- bash shell
- curl（用于检查服务健康）
- aws-cli（可选，用于 S3 操作测试）

### 3 步启动

```bash
# 1. 启动 RustFS S3 服务
./start_rustfs.sh &

# 2. 启动 Flink + Iceberg 环境
./deploy-flink-s3.sh start

# 3. 连接 SQL Client 进行测试
docker exec -it jobmanager ./bin/sql-client.sh
```

### 访问服务

- **Flink Web UI**: http://localhost:8081
- **RustFS S3**: http://localhost:9000
- **RustFS Console**: http://localhost:9001

---

## 📁 项目结构

```
.
├── deploy-flink-s3.sh                      # 🚀 一键启动脚本（推荐使用）
├── docker-compose-flink-iceberg-s3.yml     # 🐳 Docker Compose 配置
├── flink-iceberg-image/
│   └── Dockerfile                           # 📦 Flink + Iceberg 镜像定义
├── config/
│   └── rustfs/
│       └── default.env                      # ⚙️  RustFS 配置
├── data/
│   └── rustfs0/                             # 💾 S3 数据存储目录
├── logs/
│   └── rustfs/                              # 📋 RustFS 日志
├── start_rustfs.sh                          # 🔧 启动 RustFS 脚本
├── test_rustfs.sh                           # ✅ 测试 S3 连接脚本
├── .env.template                            # 📝 环境变量模板
├── FLINK_S3_DEPLOYMENT_GUIDE.md             # 📚 详细部署指南
├── FLINK_S3_QUICK_REFERENCE.md              # ⚡ 快速参考卡片
└── README.md                                # 📖 本文件
```

---

## 📖 文档

### 快速参考

最常用的命令和配置，适合已经熟悉环境的用户。

```bash
cat FLINK_S3_QUICK_REFERENCE.md
```

**内容**：
- 快速启动命令
- 常用 Docker 和 SQL 命令
- 常见问题快速解决
- 调试技巧

### 详细部署指南

从零开始的完整部署步骤，包含原理说明和故障排除。

```bash
cat FLINK_S3_DEPLOYMENT_GUIDE.md
```

**内容**：
- 详细的 5 步部署流程
- 文件配置详解
- 故障排除指南
- 监控和维护
- 参考资源

---

## 🔧 脚本使用

### 一键启动脚本（推荐）

```bash
# 启动所有服务
./deploy-flink-s3.sh start

# 停止所有服务
./deploy-flink-s3.sh stop

# 重启所有服务
./deploy-flink-s3.sh restart

# 查看日志
./deploy-flink-s3.sh logs

# 显示帮助
./deploy-flink-s3.sh help
```

### RustFS 脚本

```bash
# 启动 RustFS S3 服务（需要在后台运行）
./start_rustfs.sh &

# 测试 S3 连接和基本操作
./test_rustfs.sh
```

---

## 🐳 Docker 操作

### 直接使用 Docker Compose

```bash
# 启动
docker compose -f docker-compose-flink-iceberg-s3.yml up -d --build

# 查看状态
docker compose -f docker-compose-flink-iceberg-s3.yml ps

# 查看日志
docker compose -f docker-compose-flink-iceberg-s3.yml logs -f

# 停止
docker compose -f docker-compose-flink-iceberg-s3.yml down
```

### 构建镜像

```bash
# 使用默认配置构建
docker build -t flink-iceberg:latest ./flink-iceberg-image/

# 自定义版本构建
docker build \
  --build-arg FLINK_VERSION=2.2 \
  --build-arg ICEBERG_VERSION=1.11.1 \
  -t flink-iceberg:custom \
  ./flink-iceberg-image/
```

---

## 🔑 环境配置

### 环境变量

S3 凭据通过环境变量注入，避免硬编码在配置文件中：

```bash
export AWS_ACCESS_KEY_ID=n3xtchen
export AWS_SECRET_ACCESS_KEY=n3xtchen
export AWS_ENDPOINT_URL=http://localhost:9000
```

### 配置文件方式

创建 `.env` 文件自动加载环境变量：

```bash
# 从模板创建
cp .env.template .env

# 编辑 .env 文件（修改相应值）
# 启动时会自动加载
```

### 重要：添加到 .gitignore

```bash
# 防止敏感信息提交到仓库
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
```

---

## 📊 典型工作流

### 开发流程

```bash
# 1. 启动基础服务
./start_rustfs.sh &
./deploy-flink-s3.sh start

# 2. 检查服务状态
./deploy-flink-s3.sh logs

# 3. 连接 SQL Client 进行开发
docker exec -it jobmanager ./bin/sql-client.sh

# 4. 测试 S3 操作
docker exec -it jobmanager bash
aws s3 --endpoint-url=http://localhost:9000 ls

# 5. 完成后停止服务
./deploy-flink-s3.sh stop
pkill -f "rustfs server"
```

### CI/CD 集成

```bash
# 构建容器（用于持续集成）
docker build -t flink-iceberg:${VERSION} ./flink-iceberg-image/

# 推送到私有仓库
docker tag flink-iceberg:${VERSION} registry.example.com/flink-iceberg:${VERSION}
docker push registry.example.com/flink-iceberg:${VERSION}
```

---

## 🧪 测试

### 验证 S3 连接

```bash
./test_rustfs.sh
```

### 验证 Flink 集群

```bash
# 查看 Web UI
curl http://localhost:8081/v1/overview | jq .

# 查看 TaskManager
curl http://localhost:8081/v1/taskmanagers | jq .
```

### 验证 Iceberg 集成

```bash
docker exec -it jobmanager ./bin/sql-client.sh

# 在 SQL Client 中
SHOW CATALOGS;
CREATE TABLE test (id INT, name STRING);
INSERT INTO test VALUES (1, 'test');
SELECT * FROM test;
```

---

## 🔍 监控和调试

### 查看日志

```bash
# 实时查看 JobManager 日志
docker compose -f docker-compose-flink-iceberg-s3.yml logs -f jobmanager

# 查看 TaskManager 日志
docker logs taskmanager

# 查看特定错误
docker logs jobmanager | grep ERROR
```

### 容器内调试

```bash
# 进入容器
docker exec -it jobmanager bash

# 查看环境变量
env | grep AWS

# 检查插件
ls -la /opt/flink/plugins/
ls -la /opt/flink/lib/

# 测试 S3 连接
aws s3 --endpoint-url=http://localhost:9000 ls
```

### 性能监控

```bash
# 查看容器资源使用
docker stats

# 查看 Flink 指标
curl http://localhost:8081/v1/metrics | jq .
```

---

## 🛑 停止和清理

### 安全停止

```bash
# 停止 Flink 容器（推荐）
./deploy-flink-s3.sh stop

# 或手动停止
docker compose -f docker-compose-flink-iceberg-s3.yml down

# 停止 RustFS 服务
pkill -f "rustfs server"
```

### 完全清理

```bash
# 删除容器和网络
docker compose -f docker-compose-flink-iceberg-s3.yml down -v

# 删除镜像（可选）
docker rmi flink-iceberg:latest

# 清理 S3 数据（谨慎！）
rm -rf data/rustfs0/*
```

---

## 🆘 常见问题

### Q: 如何修改 S3 凭据？

**A**: 编辑 `.env` 文件或直接修改 `docker-compose-flink-iceberg-s3.yml` 中的环境变量。

### Q: 如何增加 TaskManager 数量？

**A**: 修改 docker-compose 文件中的 `replicas: 1` 为需要的数量。

### Q: Flink Web UI 显示 "Unreachable"？

**A**: 检查容器是否启动，等待 30 秒，查看日志获取更多信息。

### Q: S3 连接失败？

**A**: 确保 RustFS 正在运行，检查环境变量，查看网络连接。

### Q: 如何调整 Flink 并行度？

**A**: 修改 `FLINK_PROPERTIES` 中的 `parallelism.default` 和 `taskmanager.numberOfTaskSlots`。

---

## 📞 技术支持

### 查看日志

始终先查看日志获取详细错误信息：

```bash
./deploy-flink-s3.sh logs
```

### 查看文档

根据需要查看相应文档：

```bash
# 快速参考
cat FLINK_S3_QUICK_REFERENCE.md

# 详细指南
cat FLINK_S3_DEPLOYMENT_GUIDE.md
```

### 获取帮助

```bash
./deploy-flink-s3.sh help
./start_rustfs.sh --help
```

---

## 🔗 参考资源

- [Apache Flink 官方文档](https://flink.apache.org/docs/)
- [Apache Flink S3 文件系统](https://nightlies.apache.org/flink/flink-docs-stable/zh/docs/deployment/filesystems/s3/)
- [Apache Iceberg 官方文档](https://iceberg.apache.org/)
- [Apache Iceberg Flink 集成](https://iceberg.apache.org/flink/)
- [官方 iceberg-flink-quickstart](https://github.com/apache/iceberg/tree/main/docker/iceberg-flink-quickstart)

---

## 📝 版本信息

- **创建日期**: 2026-03-26
- **Flink 版本**: 2.2
- **Iceberg 版本**: 1.11.1
- **Hadoop 版本**: 3.4.2
- **Docker Compose 版本**: 3.8

---

## 💡 提示

- 💾 定期备份 S3 数据：`docker cp jobmanager:/data ./backup/`
- 🔒 生产环境请使用强密码和 IAM 凭据
- 📊 使用 Flink Web UI 监控任务状态
- 🐛 遇到问题时启用详细日志：`RUST_LOG=debug`

---

**祝您使用愉快！** 🎉

如有问题，请参考详细文档或查看容器日志。
