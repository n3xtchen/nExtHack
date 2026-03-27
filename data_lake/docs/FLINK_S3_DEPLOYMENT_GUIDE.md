# Flink S3 环境部署指南

## 📋 快速概览

本指南帮助您在已有 RustFS S3 服务的基础上，部署 Apache Flink + Iceberg + S3 集成环境。

### 环境信息
- **Flink 版本**: 2.2
- **Iceberg 版本**: 1.11.1
- **Hadoop 版本**: 3.4.2
- **S3 服务**: RustFS (自托管，监听 localhost:9000)
- **凭据管理**: 通过环境变量注入

---

## 🚀 部署步骤

### 第 1 步：启动 RustFS S3 服务

```bash
# 确保在项目根目录
cd /Users/nextchen/Dev/project_pig/nExtHack/data_lake

# 启动 RustFS 服务
./start_rustfs.sh &

# 或者先测试连接
./test_rustfs.sh
```

**预期输出**：
- 🔗 API 端点: `http://localhost:9000`
- 🔐 Access Key: `n3xtchen`
- 🔑 Secret Key: `n3xtchen`

### 第 2 步：准备 S3 环境变量

在启动 Docker 容器前，导出 S3 环境变量：

```bash
# 从 RustFS 服务获取配置
export AWS_ACCESS_KEY_ID=n3xtchen
export AWS_SECRET_ACCESS_KEY=n3xtchen
export AWS_ENDPOINT_URL=http://localhost:9000
```

也可以创建一个 `.env` 文件自动加载：

```bash
cat > .env << 'EOF'
AWS_ACCESS_KEY_ID=n3xtchen
AWS_SECRET_ACCESS_KEY=n3xtchen
AWS_ENDPOINT_URL=http://localhost:9000
EOF
```

### 第 3 步：构建 Flink Iceberg 镜像

```bash
# 使用默认配置构建
docker build -t flink-iceberg:latest ./flink-iceberg-image/

# 或者构建时指定版本
docker build \
  --build-arg FLINK_VERSION=2.2 \
  --build-arg ICEBERG_VERSION=1.11.1 \
  -t flink-iceberg:latest \
  ./flink-iceberg-image/
```

**构建过程会**：
- 下载并安装 Iceberg Flink runtime
- 下载并安装 Iceberg AWS bundle
- 下载并安装 Hadoop 客户端依赖
- 激活 S3 文件系统插件（Presto 优先，备选 Hadoop）

### 第 4 步：启动 Flink 和 Iceberg 容器

```bash
# 启动 Docker 容器
docker compose -f docker-compose-flink-iceberg-s3.yml up -d --build

# 验证容器状态
docker compose -f docker-compose-flink-iceberg-s3.yml ps
```

**预期输出**：
```
NAME          STATUS              PORTS
jobmanager    Up (healthy)        0.0.0.0:8081->8081/tcp
taskmanager   Up                  -
```

### 第 5 步：验证部署

#### 5.1 检查 Web UI

打开浏览器访问：**http://localhost:8081**

- ✅ JobManager 应该显示为 "RUNNING" 状态
- ✅ 1 个 TaskManager 应该已连接
- ✅ 检查日志中是否有 S3 相关的警告或错误

#### 5.2 检查日志中的 S3 插件

```bash
# 查看 JobManager 日志
docker logs jobmanager | grep -i s3

# 查看完整日志
docker logs jobmanager

# 实时查看日志
docker compose -f docker-compose-flink-iceberg-s3.yml logs -f jobmanager
```

**正常情况下应该看到**：
```
INFO ... S3FileSystem initialized with ...
INFO ... S3FileSystemFactory registered successfully
```

#### 5.3 连接 SQL Client 进行测试

```bash
# 连接 SQL Client
docker exec -it jobmanager ./bin/sql-client.sh
```

在 SQL Client 中执行测试命令：

```sql
-- 显示表名空间配置
SHOW CATALOGS;

-- 创建 S3 上的测试表
CREATE TABLE test_table (
  id INT,
  name STRING,
  PRIMARY KEY (id) NOT ENFORCED
)
WITH (
  'connector' = 'iceberg',
  'warehouse' = 's3://test-warehouse/',
  'catalog-type' = 'hadoop'
);

-- 插入测试数据
INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob');

-- 查询数据
SELECT * FROM test_table;

-- 退出 SQL Client
EXIT;
```

#### 5.4 直接测试 S3 访问

```bash
# 在 TaskManager 中执行 S3 操作
docker exec -it taskmanager bash

# 在容器内测试 S3 连接
aws s3 --endpoint-url=http://localhost:9000 ls
aws s3 --endpoint-url=http://localhost:9000 mb s3://flink-test
aws s3 --endpoint-url=http://localhost:9000 ls s3://flink-test

# 创建测试文件
echo "test data" > /tmp/test.txt
aws s3 --endpoint-url=http://localhost:9000 cp /tmp/test.txt s3://flink-test/

# 验证文件
aws s3 --endpoint-url=http://localhost:9000 ls s3://flink-test/

exit
```

---

## 📁 文件结构

部署完成后，您的项目结构应该如下：

```
data_lake/
├── docker-compose-flink-iceberg-s3.yml  # Flink + Iceberg 容器编排配置
├── flink-iceberg-image/
│   └── Dockerfile                       # Flink + Iceberg 镜像定义
├── config/
│   └── rustfs/
│       └── default.env                  # RustFS 配置
├── data/
│   └── rustfs0/                         # RustFS 数据存储目录
├── logs/
│   └── rustfs/                          # RustFS 日志目录
├── start_rustfs.sh                      # 启动 RustFS 脚本
└── test_rustfs.sh                       # 测试 S3 连接脚本
```

---

## 🔧 配置详解

### docker-compose-flink-iceberg-s3.yml

**关键配置项**：

| 配置 | 说明 | 值 |
|------|------|-----|
| `FLINK_VERSION` | Flink 版本 | `2.2` |
| `ICEBERG_VERSION` | Iceberg 版本 | `1.11.1` |
| `AWS_ACCESS_KEY_ID` | S3 访问密钥 ID | `n3xtchen` |
| `AWS_SECRET_ACCESS_KEY` | S3 访问密钥 | `n3xtchen` |
| `AWS_ENDPOINT_URL` | S3 服务端点 | `http://localhost:9000` |
| `taskmanager.numberOfTaskSlots` | 每个 TaskManager 的槽位数 | `2` |
| `parallelism.default` | 默认并行度 | `2` |

### Dockerfile 构建参数

```dockerfile
# 默认值（可覆盖）
ARG FLINK_VERSION=2.2
ARG ICEBERG_FLINK_RUNTIME_VERSION=2.2
ARG ICEBERG_VERSION=1.11.1
ARG HADOOP_VERSION=3.4.2
```

**自定义构建**：
```bash
docker build \
  --build-arg FLINK_VERSION=2.2 \
  --build-arg ICEBERG_VERSION=1.11.1 \
  -t flink-iceberg:custom \
  ./flink-iceberg-image/
```

---

## 🐛 故障排除

### 问题 1：容器无法启动

```bash
# 查看详细错误信息
docker compose -f docker-compose-flink-iceberg-s3.yml logs jobmanager

# 检查 Docker 磁盘空间
docker system df

# 清理未使用的镜像和容器
docker system prune -a
```

### 问题 2：S3 连接失败

```bash
# 确保 RustFS 服务正在运行
ps aux | grep rustfs

# 测试本地 S3 连接
./test_rustfs.sh

# 在容器内测试连接
docker exec -it jobmanager curl -v http://localhost:9000
```

### 问题 3：权限问题

```bash
# 检查 S3 凭据是否正确设置
docker exec jobmanager env | grep AWS

# 验证环境变量
docker compose -f docker-compose-flink-iceberg-s3.yml config | grep AWS
```

### 问题 4：Iceberg 插件未加载

```bash
# 检查镜像中的 JAR 文件
docker exec jobmanager ls -la /opt/flink/lib/iceberg/

# 检查 S3 插件
docker exec jobmanager ls -la /opt/flink/plugins/

# 查看完整构建日志
docker build --no-cache -t flink-iceberg:debug ./flink-iceberg-image/
```

---

## 📊 监控和维护

### 1. 监控 Flink 集群状态

```bash
# 查看集群概览
curl http://localhost:8081/v1/overview

# 查看正在运行的 Job
curl http://localhost:8081/v1/jobs

# 查看 TaskManager 信息
curl http://localhost:8081/v1/taskmanagers
```

### 2. 检查日志

```bash
# JobManager 日志
docker logs jobmanager

# TaskManager 日志
docker logs taskmanager

# 实时查看
docker compose -f docker-compose-flink-iceberg-s3.yml logs -f

# 查看特定错误
docker logs jobmanager | grep ERROR
```

### 3. 监控 RustFS S3 服务

```bash
# 检查 RustFS 进程
ps aux | grep rustfs

# 查看 RustFS 日志
tail -f logs/rustfs/rustfs.log

# 检查 S3 存储使用
du -sh data/rustfs0/
```

---

## 🛑 停止和清理

### 停止 Flink + Iceberg 容器

```bash
docker compose -f docker-compose-flink-iceberg-s3.yml down

# 同时删除卷
docker compose -f docker-compose-flink-iceberg-s3.yml down -v

# 同时删除镜像
docker compose -f docker-compose-flink-iceberg-s3.yml down --rmi local
```

### 停止 RustFS S3 服务

```bash
# 查看进程
ps aux | grep rustfs

# 杀死进程
kill <rustfs-pid>

# 或使用 pkill
pkill -f "rustfs server"
```

### 完全清理

```bash
# 停止所有容器和服务
docker compose -f docker-compose-flink-iceberg-s3.yml down -v
pkill -f "rustfs server"

# 清理镜像（可选）
docker rmi flink-iceberg:latest

# 清理数据（可选，谨慎操作）
rm -rf data/rustfs0/*
```

---

## 📚 参考资源

- [Apache Flink 官方文档](https://flink.apache.org/docs/)
- [Apache Flink S3 文件系统](https://nightlies.apache.org/flink/flink-docs-stable/zh/docs/deployment/filesystems/s3/)
- [Apache Iceberg 官方文档](https://iceberg.apache.org/)
- [Apache Iceberg Flink 集成](https://iceberg.apache.org/flink/)
- [iceberg-flink-quickstart GitHub](https://github.com/apache/iceberg/tree/main/docker/iceberg-flink-quickstart)

---

## ✅ 部署检查清单

- [ ] RustFS S3 服务正在运行 (`./start_rustfs.sh`)
- [ ] S3 环境变量已设置 (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`)
- [ ] Flink Iceberg 镜像已构建 (`docker build ...`)
- [ ] Docker 容器已启动 (`docker compose up -d`)
- [ ] JobManager 处于 RUNNING 状态（Web UI: http://localhost:8081）
- [ ] TaskManager 已连接
- [ ] S3 插件已加载（查看日志）
- [ ] SQL Client 可连接 (`docker exec -it jobmanager ./bin/sql-client.sh`)
- [ ] S3 读写操作正常（测试 SQL 或 CLI）
- [ ] 日志中无错误信息

---

## 📞 常见问题

**Q: 如何修改 S3 凭据？**
A: 编辑 `docker-compose-flink-iceberg-s3.yml` 中的环境变量，或设置 `.env` 文件。

**Q: 如何增加 TaskManager 数量？**
A: 修改 docker-compose 中的 `replicas: 1` 为需要的数量。

**Q: 如何持久化 Flink 数据？**
A: 添加 volumes 挂载到 docker-compose.yml，指向本地目录。

**Q: Iceberg 表存储在哪里？**
A: 默认存储在 S3 上，路径由 `CATALOG_WAREHOUSE` 或应用中的 `warehouse` 参数定义。
