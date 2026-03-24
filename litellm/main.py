import subprocess
import os
import sys

def run_proxy():
    """Run litellm proxy with the local config.yaml"""
    # Locate config.yaml relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")

    # If not found in script dir, check current working directory
    if not os.path.exists(config_path):
        config_path = os.path.join(os.getcwd(), "config.yaml")

    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}")
        sys.exit(1)

    cmd = ["litellm", "--config", config_path]
    # Add any extra args passed to the script
    cmd.extend(sys.argv[1:])

    print(f"Starting LiteLLM proxy with config: {config_path}")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nProxy stopped.")
    except Exception as e:
        print(f"Error running proxy: {e}")
        sys.exit(1)

def main():
    print("nLiteLLM Proxy CLI")
    print("Available commands:")
    print("  nlitellm-proxy  - Start the proxy server")
    print("  nlitellm-update - Update the model list in config.yaml")

if __name__ == "__main__":
    main()
