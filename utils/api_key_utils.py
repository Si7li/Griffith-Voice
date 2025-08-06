import os
import json

CONFIG_FILE = "configs/config_api.json"

def load_api_key():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            return data.get("gemini_token")
    return None

def save_api_key(api: str):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"gemini_token": api}, f)