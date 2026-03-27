# ✅ Flink S3 环境部署检查清单

**部署日期**: 2026-03-26  
**版本**: Flink 2.2 + Iceberg 1.11.1 + RustFS S3  
**状态**: ✅ 已完成

---

## 📋 部署完成确认

### ✅ 配置文件

- [x] `docker-compose-flink-iceberg-s3.yml` - 容器编排配置
- [x] `flink-iceberg-image/Dockerfile` - 镜像构建定义
- [x] `.env.template` - 环境变量模板
- [x] `config/rustfs/default.env` - RustFS 配置（已存在）

### ✅ 脚本文件

- [x] `deploy-flink-s3.sh` - 一键启动脚本（可执行）
- [x] `start_rustfs.sh` - RustFS 启动脚本（已存在）
- [x] `test_rustfs.sh` - S3 测试脚本（已存在）

### ✅ 文档文件

- [x] `README.md` - 项目总览
- [x] `FLINK_S3_QUICK_REFERENCE.md` - 快速参考
- [x] `FLINK_S3_DEPLOYMENT_GUIDE.md` - 详细指南
- [x] `FLINK_S3_DEPLOYMENT_SUMMARY.md` - 部署总结
- [x] `DEPLOYMENT_CHECKLIST.md` - 本文件

---

## 🚀 快速启动验证

### 第一步：启动 RustFS S3

```bash
# ✅ 已配置，准备就绪
./start_rustfs.sh &
```

**预期输出**:
- ✅ "RustFS 服务已启动"
- ✅ "🔗 API 端点: ... :9000"

### 第二步：启动 Flink + Iceberg

```bash
# ✅ 已配置，准备就绪
./deploy-flink-s3.sh start
```

**预期行为**:
- ✅ Docker 镜像自动构建
- ✅ 容器自动启动
- ✅ JobManager 健康检查通过

### 第三步：验证服务

```bash
# ✅ 已配置，准备就绪
curl http://localhost:8081/v1/overview
```

**预期输出**:
- ✅ JSON 格式的 Flink 集群信息
- ✅ taskmanagers 数量为 1

---

## 📁 目录结构验证

```
✅ /Users/nextchen/Dev/project_pig/nExtHack/data_lake/
  ├── ✅ docker-compose-flink-iceberg-s3.yml
  ├── ✅ flink-iceberg-image/
  │   └── ✅ Dockerfile
  ├── ✅ deploy-flink-s3.sh (可执行)
  ├── ✅ .env.template
  ├── ✅ README.md
  ├── ✅ FLINK_S3_QUICK_REFERENCE.md
  ├── ✅ FLINK_S3_DEPLOYMENT_GUIDE.md
  ├── ✅ FLINK_S3_DEPLOYMENT_SUMMARY.md
  ├── ✅ DEPLOYMENT_CHECKLIST.md (本文件)
  ├── ✅ start_rustfs.sh
  ├── ✅ test_rustfs.sh
  ├── ✅ config/rustfs/default.env
  ├── ✅ data/rustfs0/ (数据目录)
  └── ✅ logs/rustfs/ (日志目录)
```

---

## 🔑 环境变量配置

### ✅ 默认环境变量（已配置）

```bash
AWS_ACCESS_KEY_ID=n3xtchen
AWS_SECRET_ACCESS_KEY=n3xtchen
AWS_ENDPOINT_URL=http://localhost:9000
AWS_REGION=us-east-1
```

### 📝 如需自定义

```bash
# 从模板创建
cp .env.template .env

# 编辑文件
nano .env

# 验证加载
docker compose config | grep AWS
```

---

## 🧪 功能检查清单

### 在执行以下步骤前，确保已启动服务

```bash
./start_rustfs.sh &      # 启动 RustFS
./deploy-flink-s3.sh start  # 启动 Flink
sleep 30  # 等待启动完成
```

### ✅ 可以进行的测试

#### 1. S3 连接测试
```bash
# ✅ 已配置
./test_rustfs.sh
```

**验证点**:
- [ ] 命令执行成功
- [ ] 显示 S3 bucket 列表
- [ ] 能创建、上传、删除 bucket

#### 2. Flink Web UI 测试
```bash
# ✅ 已配置
curl http://localhost:8081/v1/overview | jq .
```

**验证点**:
- [ ] 返回 JSON 数据
- [ ] "taskmanagers" 数量为 1
- [ ] "available-task-slots" > 0

#### 3. SQL Client 连接测试
```bash
# ✅ 已配置
docker exec -it jobmanager ./bin/sql-client.sh
```

**在 SQL Client 内执行**:
```sql
SHOW CATALOGS;
SHOW DATABASES;
EXIT;
```

**验证点**:
- [ ] SQL 命令执行成功
- [ ] 显示 catalog 列表

#### 4. Iceberg 集成测试
```bash
# ✅ 已配置
docker exec -it jobmanager ./bin/sql-client.sh
```

**在 SQL Client 内执行**:
```sql
CREATE TABLE test_table (
  id INT NOT NULL,
  name STRING,
  PRIMARY KEY (id) NOT ENFORCED
) WITH (
  'connector' = 'iceberg',
  'warehouse' = 's3://test-warehouse/',
  'catalog-type' = 'hadoop'
);

INSERT INTO test_table VALUES (1, 'Alice');
SELECT * FROM test_table;
DROP TABLE test_table;
EXIT;
```

**验证点**:
- [ ] 表创建成功
- [ ] 数据插入成功
- [ ] 数据查询成功
- [ ] 表删除成功

#### 5. 容器日志检查
```bash
# ✅ 已配置
./deploy-flink-s3.sh logs
```

**验证点**:
- [ ] 查看到 JobManager 启动日志
- [ ] 看不到 ERROR 级别的日志
- [ ] 看到 S3 相关的初始化信息

---

## 🛑 停止和清理

### ✅ 优雅停止（推荐）

```bash
# 停止 Flink 容器
./deploy-flink-s3.sh stop

# 停止 RustFS 服务
pkill -f "rustfs server"

# 验证停止
docker ps
ps aux | grep rustfs
```

### 🧹 完全清理（仅在需要时）

```bash
# 删除容器和卷
docker compose -f docker-compose-flink-iceberg-s3.yml down -v

# 删除镜像（可选）
docker rmi flink-iceberg:latest

# 清理 S3 数据（谨慎！）
rm -rf data/rustfs0/*

# 清理日志（可选）
rm -rf logs/rustfs/*
```

---

## 🆘 常见问题排查

### 问题 1：Docker 镜像构建失败

```bash
# 检查
docker build -t flink-iceberg:latest ./flink-iceberg-image/

# 常见原因
- 网络连接问题（下载 JAR 文件）
- 磁盘空间不足
- Docker daemon 未运行

# 解决方案
1. 检查网络连接
2. 清理 Docker：docker system prune -a
3. 重启 Docker daemon
4. 再次尝试构建
```

### 问题 2：RustFS 无法启动

```bash
# 检查
./start_rustfs.sh

# 常见原因
- 端口 9000 被占用
- 数据目录权限问题
- RustFS 二进制文件不存在

# 解决方案
1. 检查端口：lsof -i :9000
2. 检查数据目录权限：ls -la data/
3. 检查配置文件：cat config/rustfs/default.env
4. 杀死已有进程：pkill -f rustfs
5. 重新启动：./start_rustfs.sh
```

### 问题 3：Flink JobManager 无法就绪

```bash
# 检查
curl http://localhost:8081/v1/overview
./deploy-flink-s3.sh logs

# 常见原因
- 容器启动时间过长
- 网络配置问题
- 内存不足

# 解决方案
1. 等待更久（最多 60 秒）
2. 检查 Docker 网络：docker network ls
3. 检查系统资源：docker stats
4. 查看完整日志：docker logs jobmanager
```

### 问题 4：S3 连接失败

```bash
# 检查
docker exec -it jobmanager bash
aws s3 --endpoint-url=http://localhost:9000 ls

# 常见原因
- RustFS 服务未运行
- 端点地址错误
- 凭据配置不正确

# 解决方案
1. 启动 RustFS：./start_rustfs.sh
2. 验证端点：ping localhost:9000
3. 检查环境变量：env | grep AWS
4. 运行测试脚本：./test_rustfs.sh
```

---

## 📊 性能基线

### 系统要求

```
✅ CPU: 2+ 核心
✅ 内存: 4GB (建议 8GB)
✅ 磁盘: 20GB+ (用于镜像、容器、S3 数据)
✅ 网络: 100Mbps+ (用于 S3 操作)
```

### 默认配置性能

```yaml
JobManager:
  内存: ~1GB
  CPU: 1 核心共享

TaskManager x1:
  内存: ~1.5GB
  CPU: 2 逻辑核心
  任务槽位: 2

RustFS:
  内存: ~500MB
  CPU: 1 核心共享

总计:
  内存占用: ~3GB (建议 6GB+)
  CPU 占用: ~2 核心
```

---

## 🔐 安全配置

### ✅ 开发环境（当前）

```bash
✅ S3 凭据: n3xtchen / n3xtchen (简单凭据)
✅ 网络访问: 仅本地 (localhost)
✅ 加密: 无 (HTTP)
✅ 认证: 基于凭据
```

### ⚠️ 生产迁移前

```bash
⚠️ 必须 修改 S3 凭据为强密码
⚠️ 必须 启用 TLS/HTTPS
⚠️ 必须 配置防火墙规则
⚠️ 必须 启用 IAM 和访问控制
⚠️ 必须 添加审计日志
⚠️ 禁止 将 .env 提交到 Git
✅ 推荐 使用 .env.local 管理本地凭据
✅ 推荐 设置 .gitignore 规则
```

---

## 📞 获取帮助

### 查看文档

```bash
# 快速开始
cat README.md

# 快速参考
cat FLINK_S3_QUICK_REFERENCE.md

# 详细指南
cat FLINK_S3_DEPLOYMENT_GUIDE.md

# 部署总结
cat FLINK_S3_DEPLOYMENT_SUMMARY.md
```

### 查看日志

```bash
# 实时日志
./deploy-flink-s3.sh logs

# 完整日志
docker logs jobmanager

# 特定错误
docker logs jobmanager | grep ERROR
```

### 测试连接

```bash
# S3 连接
./test_rustfs.sh

# Flink 集群
curl http://localhost:8081/v1/overview | jq .

# RustFS 服务
curl http://localhost:9000/
```

---

## 📝 后续步骤

### 短期 (第 1 周)
- [ ] 启动所有服务并验证
- [ ] 阅读 README.md 和 QUICK_REFERENCE.md
- [ ] 进行基本的 SQL 操作测试
- [ ] 理解 Flink Web UI

### 中期 (第 2-4 周)
- [ ] 创建自定义 Iceberg 表
- [ ] 编写 SQL 查询和转换
- [ ] 学习 Flink SQL 高级特性
- [ ] 配置 checkpoint (可选)

### 长期 (第 5+ 周)
- [ ] 开发生产级 Flink Job
- [ ] 部署多个 TaskManager
- [ ] 建立监控和告警
- [ ] 优化性能和成本

---

## 🎉 完成！

**恭喜！您的 Flink S3 环境已部署完成！**

### 立即开始

```bash
# 1. 启动服务
./start_rustfs.sh &
./deploy-flink-s3.sh start

# 2. 打开浏览器
# Flink Web UI: http://localhost:8081

# 3. 连接 SQL Client
docker exec -it jobmanager ./bin/sql-client.sh

# 4. 开始开发！
```

### 常用快捷方式

```bash
# 启动
alias flink_start='./start_rustfs.sh & sleep 5 && ./deploy-flink-s3.sh start'

# 停止
alias flink_stop='./deploy-flink-s3.sh stop && pkill -f "rustfs server"'

# 查看日志
alias flink_logs='./deploy-flink-s3.sh logs'

# SQL Client
alias flink_sql='docker exec -it jobmanager ./bin/sql-client.sh'

# 测试 S3
alias flink_test='./test_rustfs.sh'
```

---

**问题或建议？** 参考相应文档或查看日志获取更多信息。

**祝您使用愉快！** 🚀
