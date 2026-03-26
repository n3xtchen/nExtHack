#!/bin/bash

# Flink + Iceberg 写入 S3 的测试脚本
# 使用 Iceberg 作为数据湖框架，通过 S3 作为存储后端

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Flink + Iceberg + S3 数据湖测试${NC}"
echo -e "${BLUE}════════════════════════════════════════════${NC}"

# 步骤 0: 验证环境
echo ""
echo -e "${BLUE}[0/3] 验证环境配置...${NC}"

echo "检查 S3 Presto 插件..."
PLUGIN_COUNT=$(docker exec jobmanager ls -1 /opt/flink/plugins/s3-fs-presto/ 2>/dev/null | wc -l)
if [ "$PLUGIN_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ S3 Presto 插件已激活${NC}"
else
    echo -e "${RED}❌ S3 Presto 插件未激活${NC}"
fi

echo "检查 Iceberg 支持..."
docker exec jobmanager find /opt/flink/lib -name "*iceberg*" 2>/dev/null | wc -l > /tmp/iceberg_count.txt
ICEBERG_JARS=$(cat /tmp/iceberg_count.txt)
if [ "$ICEBERG_JARS" -gt 0 ]; then
    echo -e "${GREEN}✅ Iceberg 库已安装${NC}"
else
    echo -e "${YELLOW}⚠️  未检测到 Iceberg 库${NC}"
fi

# 步骤 1: 创建 Iceberg 表
echo ""
echo -e "${BLUE}[1/3] 创建 Iceberg 表...${NC}"

docker exec -i jobmanager ./bin/sql-client.sh 2>&1 << 'SQL_END' > /tmp/iceberg_output.txt
-- 创建 Iceberg 表（使用 S3 作为存储）
CREATE TABLE IF NOT EXISTS iceberg_users (
  id INT,
  name STRING,
  age INT,
  city STRING
) WITH (
  'connector' = 'iceberg',
  'warehouse' = 's3://flink-data/iceberg/',
  'uri' = 's3://flink-data/iceberg/',
  'format' = 'parquet'
);

-- 插入数据
INSERT INTO iceberg_users VALUES
  (1, 'Alice', 30, 'Beijing'),
  (2, 'Bob', 25, 'Shanghai'),
  (3, 'Charlie', 35, 'Guangzhou'),
  (4, 'Diana', 28, 'Shenzhen');
SQL_END

if grep -q "Job ID:" /tmp/iceberg_output.txt; then
    echo -e "${GREEN}✅ Iceberg 表创建成功${NC}"
    echo -e "${GREEN}✅ INSERT 提交成功${NC}"
    ICEBERG_JOB_ID=$(grep "Job ID:" /tmp/iceberg_output.txt | head -1 | awk '{print $NF}')
    echo "Job ID: $ICEBERG_JOB_ID"
else
    echo -e "${RED}❌ Iceberg 表操作失败${NC}"
    cat /tmp/iceberg_output.txt | tail -50
fi

# 等待数据写入
sleep 5

# 步骤 2: 查看 S3 数据
echo ""
echo -e "${BLUE}[2/3] 查看 S3 中的 Iceberg 数据...${NC}"

export AWS_ACCESS_KEY_ID=n3xtchen
export AWS_SECRET_ACCESS_KEY=n3xtchen

echo ""
echo "S3 Iceberg 目录内容:"
aws s3 --endpoint-url=http://localhost:9000 ls s3://flink-data/iceberg/ --recursive 2>&1 | head -30

echo ""
echo "查找 Parquet 文件:"
aws s3 --endpoint-url=http://localhost:9000 ls s3://flink-data/iceberg/ --recursive 2>/dev/null | grep -E "\.parquet$" && ICEBERG_HAS_DATA=true || ICEBERG_HAS_DATA=false

# 步骤 3: 显示结论
echo ""
echo -e "${BLUE}[3/3] 结论${NC}"
echo ""

if [ "$ICEBERG_HAS_DATA" = true ]; then
    echo -e "${GREEN}✅ Iceberg + S3 数据湖写入成功！${NC}"
    echo "数据已成功写入到 S3 的 Iceberg 格式"
else
    echo -e "${RED}❌ Iceberg 数据未写入 S3${NC}"
    echo ""
    echo -e "${YELLOW}可能的原因:${NC}"
    echo "1. Iceberg 连接器不可用或未正确配置"
    echo "2. S3 连接问题"
    echo "3. Flink 存在 Iceberg 相关的 BUG"
    echo ""
    echo -e "${YELLOW}查看完整日志:${NC}"
    echo "docker logs jobmanager | grep -i iceberg | tail -20"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════${NC}"
