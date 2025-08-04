import os
import json

CONFIG_FILE = "configs/config.json"

def load_token():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            return data.get("hf_token")
    return None

def save_token(token: str):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"hf_token": token}, f)