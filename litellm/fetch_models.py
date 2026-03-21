import os
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

def fetch_gemini_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env")
        return []

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        models = []
        for m in data.get('models', []):
            name = m['name'].replace('models/', '')
            # Only include models that support generateContent
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                models.append({
                    'model_name': f'gemini/{name}',
                    'litellm_params': {
                        'model': f'gemini/{name}',
                        'api_key': 'os.environ/GEMINI_API_KEY'
                    }
                })
        return models
    except Exception as e:
        print(f"Error fetching Gemini models: {e}")
        return []

def fetch_copilot_models():
    token = os.getenv("HOMEBREW_GITHUB_API_TOKEN")
    if not token:
        print("Error: HOMEBREW_GITHUB_API_TOKEN not found in environment")
        return []

    # Try fetching from https://api.github.com/copilot/models
    # This often requires specific headers
    urls = [
        "https://api.github.com/copilot/models",
        "https://api.githubcopilot.com/models"
    ]
    headers = {
        "Authorization": f"Bearer {token}",
        "Editor-Version": "vscode/1.97.2",
        "Editor-Plugin-Version": "copilot-chat/0.25.2025012401",
        "User-Agent": "GitHubCopilotChat/0.25.2025012401",
        "Accept": "application/json"
    }

    for url in urls:
        try:
            print(f"Trying Copilot URL: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = []
                # Copilot models response format is usually a list of models in 'data' or root
                model_list = data if isinstance(data, list) else data.get('data', [])
                for m in model_list:
                    m_id = m.get('id')
                    if not m_id:
                        continue
                    models.append({
                        'model_name': f'gh{m_id}',
                        'litellm_params': {
                            'model': f'github_copilot/{m_id}',
                            'extra_headers': {
                                "Editor-Version": "vscode/1.85.1",
                                "Copilot-Integration-Id": "vscode-chat"
                            }
                        }
                    })
                return models
            else:
                print(f"Copilot API ({url}) returned {response.status_code}")
        except Exception as e:
            print(f"Error fetching from {url}: {e}")

    # If all failed, check if we can get a token first
    try:
        print("Trying to get a temporary copilot token...")
        token_url = "https://api.github.com/copilot_internal/v2/token"
        resp = requests.get(token_url, headers={"Authorization": f"token {token}"}, timeout=10)
        print(f"Token request status: {resp.status_code}")
        if resp.status_code == 200:
            temp_token_data = resp.json()
            temp_token = temp_token_data.get('token')
            if temp_token:
                print("Successfully got temporary token.")
                headers["Authorization"] = f"Bearer {temp_token}"
                url = "https://api.githubcopilot.com/models"
                print(f"Trying with temporary token: {url}")
                response = requests.get(url, headers=headers, timeout=10)
                print(f"Models request with temp token status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    model_list = data if isinstance(data, list) else data.get('data', [])
                    models = []
                    for m in model_list:
                        m_id = m.get('id')
                        if not m_id: continue
                        models.append({
                            'model_name': f'gh{m_id}',
                            'litellm_params': {
                                'model': f'github_copilot/{m_id}',
                                'extra_headers': {
                                    "Editor-Version": "vscode/1.85.1",
                                    "Copilot-Integration-Id": "vscode-chat"
                                }
                            }
                        })
                    return models
                else:
                    print(f"Models request failed: {response.text}")
        else:
            print(f"Token request failed: {resp.text}")
    except Exception as e:
        print(f"Error getting temp token: {e}")

    # Fallback to standard models if scanning fails
    print("Scanning Copilot failed, using standard fallback models.")
    standard_models = ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo', 'claude-3.5-sonnet', 'o1', 'o1-mini']
    models = []
    for m_id in standard_models:
        models.append({
            'model_name': f'gh{m_id}',
            'litellm_params': {
                'model': f'github_copilot/{m_id}',
                'extra_headers': {
                    "Editor-Version": "vscode/1.85.1",
                    "Copilot-Integration-Id": "vscode-chat"
                }
            }
        })
    return models

def main():
    print("Fetching Gemini models...")
    gemini_models = fetch_gemini_models()
    print(f"Found {len(gemini_models)} Gemini models.")

    print("Fetching Copilot models...")
    copilot_models = fetch_copilot_models()
    print(f"Found {len(copilot_models)} Copilot models.")

    config = {
        'model_list': gemini_models + copilot_models
    }

    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

    print("Successfully generated config.yaml")

if __name__ == "__main__":
    main()
