#!/bin/bash

root_dir="$(pwd)"
config_env="${root_dir}/config/rustfs/default.env"
data_dir="${root_dir}/data"
logs_dir="${root_dir}/logs/rustfs"

# 检查配置
if [ ! -f "${config_env}" ]; then
    echo "❌ 配置文件不存在，请先创建 ${config_env}"
    exit 1
fi

set -a
source "${config_env}"
RUSTFS_OBS_LOG_DIRECTORY="${logs_dir}"
set +a

rustfs server "${data_dir}"

echo "✅ RustFS服务已启动"
echo "📊 控制台: ${RUSTFS_CONSOLE_ADDRESS}"
echo "🔗 API端点: ${RUSTFS_ADDRESS}"
