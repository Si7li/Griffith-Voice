import os
import json

def save_cache(cache_path, data):
    """Save data to cache file"""
    if cache_path:
        try:
            # Create directory if it doesn't exist
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Cache saved to: {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

def read_cache(read_from_cache, cache_path):
    """Read data from cache file if it exists and cache is enabled"""
    if read_from_cache and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            print(f"Using cached data from: {cache_path}")
            return data
        except Exception as e:
            print(f"Failed to read cache: {e}")
    return None
