import subprocess
import sys
import os
import shutil
from utils import save_cache, read_cache

class SeparateAudio:
    def __init__(self, input_audio):
        self.input_audio = input_audio
    def separate_audio(self,read_from_cache=False, cache_path=None):
        print(f"Separating audio: {self.input_audio}")
        paths = read_cache(read_from_cache, cache_path)
        if paths:
            return paths

        print("This may take a few minutes...")
        
        temp_output = "temp_separation"
        cmd = [
            sys.executable, "-m", "demucs.separate",
            "-n", "htdemucs",
            "--two-stems", "vocals",
            "-o", temp_output,
            self.input_audio
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Separation completed!")
            
            # Get the original filename and output directory
            filename = os.path.splitext(os.path.basename(self.input_audio))[0]
            output_dir = os.path.dirname(self.input_audio)
            
            # Demucs creates files in: temp_output/htdemucs/filename/vocals.wav and no_vocals.wav
            demucs_output_dir = f"{temp_output}/htdemucs/{filename}"
            source_vocals = f"{demucs_output_dir}/vocals.wav"
            source_music = f"{demucs_output_dir}/no_vocals.wav"
            
            # Define target paths in the outputs folder
            target_vocals = f"{output_dir}/vocals.wav"
            target_music = f"{output_dir}/no_vocals.wav"
            
            # Move files to the desired location
            if os.path.exists(source_vocals):
                shutil.move(source_vocals, target_vocals)
            else:
                print(f"Warning: Expected vocals file not found at {source_vocals}")
                
            if os.path.exists(source_music):
                shutil.move(source_music, target_music)
            else:
                print(f"Warning: Expected music file not found at {source_music}")
            
            # Clean up temporary directory
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)
            
            paths = {
                "vocals": target_vocals,
                "music": target_music,
                "output_dir": output_dir
            }
            # Save cache
            if cache_path:
                save_cache(cache_path, paths)

            return paths
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None