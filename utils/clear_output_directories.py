import os

def clear_output_directories():
    """Clear all output directories before processing a new video"""
    import shutil
    
    output_dirs = [
        "outputs",
        "outputs/audio_segments", 
        "outputs/voice_samples",
        "outputs/translated_outputs"
    ]
    
    print("Clearing output directories...")
    
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            try:
                # Remove all contents but keep the directory
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                print(f"  Cleared: {dir_path}")
            except Exception as e:
                print(f"  Could not clear {dir_path}: {e}")
        else:
            # Create directory if it doesn't exist
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  Created: {dir_path}")
            except Exception as e:
                print(f"  Could not create {dir_path}: {e}")
    
    print("Output directories cleared!")