import litellm
import yaml
import os
import argparse
import shutil
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def fetch_new_model_list():
    """使用 litellm 获取最新的 Gemini 和 Copilot 模型列表"""
    new_model_list = []

    # Gemini models
    print("Scanning Gemini models...")
    gemini_models = sorted(list(litellm.models_by_provider.get('gemini', [])))
    for model in gemini_models:
        full_model_name = model if '/' in model else f"gemini/{model}"
        new_model_list.append({
            'model_name': full_model_name.split('/')[-1],
            'litellm_params': {
                'model': full_model_name,
                'api_key': 'os.environ/GEMINI_API_KEY'
            }
        })

    # Copilot models
    print("Scanning Copilot models...")
    copilot_models = sorted(list(litellm.models_by_provider.get('github_copilot', [])))
    for model in copilot_models:
        model_id = model.split('/')[-1]
        model_name = f"gh-{model_id}"
        new_model_list.append({
            'model_name': model_name,
            'litellm_params': {
                'model': f"github_copilot/{model_id}",
                'extra_headers': {
                    "Editor-Version": "vscode/1.85.1",
                    "Copilot-Integration-Id": "vscode-chat"
                }
            }
        })

    return new_model_list

def generate_yaml_string(config_data):
    """手动构建 YAML 字符串以匹配特定的缩进和单行对象格式"""
    output = ""
    # 先处理非 model_list 的键
    for key, value in config_data.items():
        if key == 'model_list':
            continue
        output += yaml.dump({key: value}, sort_keys=False, allow_unicode=True)

    # 处理 model_list
    output += "model_list:\n"
    for item in config_data.get('model_list', []):
        output += f"  - model_name: {item['model_name']}\n"
        output += f"    litellm_params:\n"
        params = item['litellm_params']
        output += f"      model: {params['model']}\n"
        if 'api_key' in params:
            output += f"      api_key: {params['api_key']}\n"
        if 'extra_headers' in params:
            h = params['extra_headers']
            headers_str = '{"Editor-Version": "' + str(h.get('Editor-Version', '')) + '", "Copilot-Integration-Id": "' + str(h.get('Copilot-Integration-Id', '')) + '"}'
            output += f"      extra_headers: {headers_str}\n"

    return output

def main():
    parser = argparse.ArgumentParser(description="LiteLLM Model List Config Updater CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to the config.yaml file (default: config.yaml)")
    parser.add_argument("--no-backup", action="store_true", help="Disable automatic backup")

    args = parser.parse_args()
    config_path = args.config

    if not os.path.exists(config_path):
        print(f"Error: File {config_path} not found.")
        return

    # 1. 备份文件
    if not args.no_backup:
        backup_path = f"{config_path}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
        print(f"Backing up {config_path} to {backup_path}...")
        shutil.copy2(config_path, backup_path)

    # 2. 读取现有配置
    print(f"Reading {config_path}...")
    with open(config_path, 'r') as f:
        try:
            config_data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error parsing YAML: {e}")
            return

    # 3. 获取新模型并替换 model_list
    new_list = fetch_new_model_list()
    config_data['model_list'] = new_list

    # 4. 生成格式化的 YAML 字符串
    formatted_output = generate_yaml_string(config_data)

    # 5. 写回文件
    with open(config_path, 'w') as f:
        f.write(formatted_output)

    print(f"\nSuccess! Updated {config_path}.")
    print(f"Replaced model_list with {len(new_list)} models while preserving other configuration keys.")

if __name__ == "__main__":
    main()
