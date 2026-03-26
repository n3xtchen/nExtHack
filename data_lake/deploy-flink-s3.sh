#!/bin/bash

##############################################################################
# Flink S3 一键启动脚本
#
# 使用：./deploy-flink-s3.sh [start|stop|restart|logs]
#
# 说明：
#   start   - 启动 Flink + Iceberg 环境（需要 RustFS 已启动）
#   stop    - 停止 Flink + Iceberg 容器
#   restart - 重启 Flink + Iceberg 容器
#   logs    - 查看 Flink 容器日志
##############################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose-flink-iceberg-s3.yml"
DOCKER_IMAGE_DIR="${SCRIPT_DIR}/flink-iceberg-image"

# 函数：打印标题
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
}

# 函数：打印成功信息
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

# 函数：打印警告信息
print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 函数：打印错误信息
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 函数：打印信息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# 函数：检查 RustFS 是否运行
check_rustfs() {
    if ! pgrep -f "rustfs server" > /dev/null; then
        print_warning "RustFS 服务未运行"
        print_info "请先启动 RustFS: ./start_rustfs.sh &"
        return 1
    fi
    print_success "RustFS 服务正在运行"
    return 0
}

# 函数：检查 Docker 是否运行
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker 未运行或无权限访问"
        exit 1
    fi
    print_success "Docker 已就绪"
}

# 函数：检查 docker-compose 是否存在
check_compose_file() {
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "docker-compose 配置文件不存在: $COMPOSE_FILE"
        exit 1
    fi
    print_success "docker-compose 配置文件已找到"
}

# 函数：启动
start() {
    print_header "启动 Flink + Iceberg + S3 环境"

    # 检查前置条件
    check_docker
    check_compose_file
    check_rustfs || exit 1

    # 设置 S3 环境变量
    print_info "配置 S3 环境变量..."
    export AWS_ACCESS_KEY_ID=n3xtchen
    export AWS_SECRET_ACCESS_KEY=n3xtchen
    export AWS_ENDPOINT_URL=http://localhost:9000
    print_success "S3 环境变量已设置"

    # 检查是否需要重新构建
    if [ "$REBUILD_IMAGE" = "1" ] || ! docker images | grep -q "flink-iceberg"; then
        print_info "开始构建 Flink Iceberg 镜像..."
        cd "$SCRIPT_DIR"
        docker compose -f "$COMPOSE_FILE" build --no-cache
        print_success "Flink Iceberg 镜像构建完成"
    else
        print_success "Flink Iceberg 镜像已存在 (使用 'rebuild' 参数强制重建)"
    fi

    # 启动容器
    print_info "启动 Flink 和 Iceberg 容器..."
    cd "$SCRIPT_DIR"
    docker compose -f "$COMPOSE_FILE" up -d
    print_success "容器已启动"

    # 等待容器就绪
    print_info "等待容器就绪... (最多 60 秒)"
    max_retries=12
    retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        if curl -s http://localhost:8081/overview > /dev/null 2>&1; then
            print_success "Flink JobManager 已就绪"
            break
        fi
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $max_retries ]; then
            print_info "等待中... ($((retry_count * 5)) 秒)"
            sleep 5
        fi
    done

    if [ $retry_count -eq $max_retries ]; then
        print_warning "Flink JobManager 启动超时，请手动检查"
        docker compose -f "$COMPOSE_FILE" ps
    fi

    # 显示容器状态
    print_header "容器状态"
    docker compose -f "$COMPOSE_FILE" ps

    # 显示访问信息
    print_header "服务访问信息"
    print_success "Flink Web UI: http://localhost:8081"
    print_success "SQL Client: docker exec -it jobmanager ./bin/sql-client.sh"
    print_success "查看日志: docker compose -f $COMPOSE_FILE logs -f jobmanager"

    print_header "下一步操作"
    echo "1. 打开浏览器访问 http://localhost:8081 查看 Flink Web UI"
    echo "2. 连接 SQL Client: docker exec -it jobmanager ./bin/sql-client.sh"
    echo "3. 查看完整部署指南: cat FLINK_S3_DEPLOYMENT_GUIDE.md"
}

# 函数：停止
stop() {
    print_header "停止 Flink + Iceberg 环境"

    check_compose_file

    if docker compose -f "$COMPOSE_FILE" ps | grep -q "jobmanager"; then
        print_info "停止容器..."
        cd "$SCRIPT_DIR"
        docker compose -f "$COMPOSE_FILE" down
        print_success "容器已停止"
    else
        print_info "容器已停止"
    fi
}

# 函数：重启
restart() {
    print_header "重启 Flink + Iceberg 环境"
    stop
    sleep 2
    start
}

# 函数：查看日志
logs() {
    check_compose_file
    print_info "显示 Flink 日志 (按 Ctrl+C 退出)..."
    cd "$SCRIPT_DIR"
    docker compose -f "$COMPOSE_FILE" logs -f jobmanager
}

# 函数：显示帮助
show_help() {
    cat << EOF
Flink S3 一键启动脚本

使用方法：
  $0 [命令]

命令：
  start       启动 Flink + Iceberg + S3 环境
  rebuild     强制重建镜像并启动
  stop        停止 Flink + Iceberg 容器
  restart     重启 Flink + Iceberg 容器
  logs        查看 Flink 容器日志
  help        显示此帮助信息

示例：
  $0 start        # 启动环境（使用缓存镜像）
  $0 rebuild      # 强制重建镜像后启动
  $0 stop         # 停止环境
  $0 restart      # 重启环境
  $0 logs         # 查看日志

前置要求：
  - Docker 已安装并运行
  - RustFS S3 服务已启动 (./start_rustfs.sh)
  - docker-compose-flink-iceberg-s3.yml 配置文件存在

常用命令：
  查看容器状态：
    docker ps

  连接 SQL Client：
    docker exec -it jobmanager ./bin/sql-client.sh

  测试 S3 连接：
    docker exec -it jobmanager bash
    aws s3 --endpoint-url=http://localhost:9000 ls

  查看完整日志：
    docker logs jobmanager

  Flink Web UI：
    http://localhost:8081

EOF
}

# 主逻辑
case "${1:-start}" in
    start)
        REBUILD_IMAGE=0
        start
        ;;
    rebuild)
        REBUILD_IMAGE=1
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    help)
        show_help
        ;;
    *)
        print_error "未知命令: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
