# Flink S3 快速参考卡片

## 🚀 快速启动（3 步）

```bash
# 1. 启动 RustFS S3 服务
./start_rustfs.sh &

# 2. 启动 Flink + Iceberg 环境
./deploy-flink-s3.sh start

# 3. 连接 SQL Client
docker exec -it jobmanager ./bin/sql-client.sh
```

## 📝 环境变量

```bash
# S3 凭据
AWS_ACCESS_KEY_ID=n3xtchen
AWS_SECRET_ACCESS_KEY=n3xtchen
AWS_ENDPOINT_URL=http://localhost:9000

# 可选：Flink 配置
FLINK_VERSION=2.2
ICEBERG_VERSION=1.11.1
HADOOP_VERSION=3.4.2
```

## 🔗 常用链接

| 服务 | URL | 说明 |
|------|-----|------|
| Flink Web UI | http://localhost:8081 | Flink 管理界面 |
| RustFS S3 | http://localhost:9000 | S3 兼容服务 |
| RustFS Console | http://localhost:9001 | RustFS 管理界面 |

## 📋 常用命令

### 启动和停止

```bash
# 启动
./deploy-flink-s3.sh start

# 停止
./deploy-flink-s3.sh stop

# 重启
./deploy-flink-s3.sh restart

# 查看日志
./deploy-flink-s3.sh logs
```

### Docker 操作

```bash
# 查看容器状态
docker compose -f docker-compose-flink-iceberg-s3.yml ps

# 查看完整日志
docker logs jobmanager

# 进入容器
docker exec -it jobmanager bash

# 停止和删除容器
docker compose -f docker-compose-flink-iceberg-s3.yml down
```

### SQL 操作

```bash
# 连接 SQL Client
docker exec -it jobmanager ./bin/sql-client.sh

# 一次性执行 SQL 命令
docker exec -it jobmanager ./bin/sql-client.sh -e "SHOW CATALOGS;"
```

### S3 测试

```bash
# 在容器内执行 S3 命令
docker exec -it jobmanager bash

# 列出 S3 bucket
aws s3 --endpoint-url=http://localhost:9000 ls

# 创建 bucket
aws s3 --endpoint-url=http://localhost:9000 mb s3://test-bucket

# 上传文件
aws s3 --endpoint-url=http://localhost:9000 cp myfile.txt s3://test-bucket/

# 删除 bucket
aws s3 --endpoint-url=http://localhost:9000 rb s3://test-bucket
```

## 🧪 测试 SQL 脚本

```sql
-- 显示所有 catalog
SHOW CATALOGS;

-- 显示数据库
SHOW DATABASES;

-- 创建表
CREATE TABLE test_table (
  id INT NOT NULL,
  name STRING,
  age INT,
  PRIMARY KEY (id) NOT ENFORCED
) WITH (
  'connector' = 'iceberg',
  'warehouse' = 's3://warehouse/',
  'catalog-type' = 'hadoop'
);

-- 插入数据
INSERT INTO test_table VALUES
  (1, 'Alice', 30),
  (2, 'Bob', 25),
  (3, 'Charlie', 35);

-- 查询数据
SELECT * FROM test_table;

-- 查看表信息
DESCRIBE test_table;

-- 删除表
DROP TABLE test_table;
```

## 🔧 配置文件

### docker-compose-flink-iceberg-s3.yml

主要配置：
- `FLINK_PROPERTIES` - Flink 运行时配置
- `AWS_ACCESS_KEY_ID` - S3 访问密钥
- `AWS_SECRET_ACCESS_KEY` - S3 访问密码
- `AWS_ENDPOINT_URL` - S3 端点

### Dockerfile

在 `flink-iceberg-image/Dockerfile` 中修改：
- `FLINK_VERSION` - Flink 版本（默认 2.2）
- `ICEBERG_VERSION` - Iceberg 版本（默认 1.11.1）
- `HADOOP_VERSION` - Hadoop 版本（默认 3.4.2）

## 📊 目录结构

```
data_lake/
├── deploy-flink-s3.sh                  # 一键启动脚本
├── docker-compose-flink-iceberg-s3.yml # 容器编排配置
├── flink-iceberg-image/
│   └── Dockerfile                       # Flink Iceberg 镜像
├── config/
│   └── rustfs/
│       └── default.env                  # RustFS 配置
├── data/
│   └── rustfs0/                         # S3 数据存储
├── logs/
│   ├── rustfs/                          # RustFS 日志
│   └── flink/                           # Flink 日志（容器内）
├── start_rustfs.sh                      # 启动 RustFS 脚本
├── test_rustfs.sh                       # 测试 S3 连接脚本
├── FLINK_S3_DEPLOYMENT_GUIDE.md         # 详细部署指南
└── FLINK_S3_QUICK_REFERENCE.md          # 本文件
```

## 🆘 常见问题快速解决

| 问题 | 解决方案 |
|------|--------|
| 容器无法启动 | 检查 Docker 是否运行，查看日志：`./deploy-flink-s3.sh logs` |
| RustFS 连接失败 | 确保 RustFS 正在运行：`./start_rustfs.sh &` |
| S3 凭据错误 | 检查环境变量：`env \| grep AWS` |
| Web UI 无法访问 | 等待容器完全启动（约 30 秒），查看状态：`docker ps` |
| Iceberg 插件未加载 | 检查构建日志，重建镜像：`docker build -t flink-iceberg:latest ./flink-iceberg-image/` |

## 📞 调试技巧

```bash
# 查看环境变量是否正确
docker exec jobmanager env | grep AWS

# 检查 S3 插件是否加载
docker exec jobmanager ls -la /opt/flink/plugins/

# 查看 Iceberg JAR 文件
docker exec jobmanager ls -la /opt/flink/lib/iceberg/

# 实时查看 JobManager 日志
docker compose -f docker-compose-flink-iceberg-s3.yml logs -f jobmanager

# 进入容器内部排查
docker exec -it jobmanager bash
```

## 🔒 安全提示

- ⚠️ 当前使用的凭据 (`n3xtchen`/`n3xtchen`) 仅用于本地开发
- ⚠️ 生产环境请使用强密码和 IAM 凭据
- ⚠️ 不要将凭据提交到版本控制系统
- ✅ 使用 `.env` 文件管理敏感信息
- ✅ 设置 `.gitignore` 排除 `.env` 文件

## 📚 更多资源

- [详细部署指南](./FLINK_S3_DEPLOYMENT_GUIDE.md)
- [Apache Flink 文档](https://flink.apache.org/docs/)
- [Apache Iceberg 文档](https://iceberg.apache.org/)
- [官方 iceberg-flink-quickstart](https://github.com/apache/iceberg/tree/main/docker/iceberg-flink-quickstart)

---

更新日期：2026-03-26
