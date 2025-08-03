import ffmpeg
import os
from utils import save_cache, read_cache

class ExtractAudio:
    def __init__(self, input_path):
        self.input_path = input_path

    def extract_audio(self, output_path, read_from_cache=False, cache_path=None):
        path = read_cache(read_from_cache, cache_path)
        if path:
            self.output_path = path
            return path
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(self.input_path)
                .output(output_path, acodec='pcm_s16le')
                .overwrite_output()
                .run()
            )
            print(f"Audio extracted to {output_path}")
            self.output_path = output_path
            # Save cache
            if cache_path:
                save_cache(cache_path, output_path)
            return output_path
        except ffmpeg.Error as e:
            print(f"An ffmpeg error occurred: {e}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None