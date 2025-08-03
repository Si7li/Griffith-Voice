from utils import load_token, save_token
from pyannote.audio import Pipeline
from collections import defaultdict
import torch
import os
from utils import save_cache, read_cache


class AudioDiarization:
    def __init__(self, vocal_input):
        self.vocal_input = vocal_input
    
    def diarize_audio(self, read_from_cache=False, cache_path=None):
        diarization = read_cache(read_from_cache, cache_path)
        if diarization:
            return diarization
        
        if not os.path.exists(self.vocal_input):
            print(f"Error: Audio file '{self.vocal_input}' not found")
            return None
        try:
            # Initialize the speaker diarization pipeline
            # Try loading token from cache
            auth_token = load_token()
            # If not found, ask user and save it
            if not auth_token:
                auth_token = input("🔐 Enter your Hugging Face token: ").strip()
                save_token(auth_token)
            
            print("Loading speaker diarization pipeline...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )
            
            print(f"Processing audio file: {self.vocal_input}")
            # Send tensors to the gpu
            pipeline.to(torch.device("cuda"))
            diarization = pipeline(self.vocal_input)
            diarization_essensials = defaultdict(list)
            print("\nSpeaker Diarization Results:")
            print("-" * 40)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"Speaker {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")
                diarization_essensials[speaker] += [(turn.start, turn.end)]
            if cache_path:
                save_cache(cache_path, diarization_essensials)
            return diarization_essensials
            
        except Exception as e:
            print(f"Error during speaker diarization: {e}")
            
            # Provide specific guidance based on the error
            if "segmentation" in str(e):
                print("\n🎯 Missing license for segmentation model:")
                print("   Visit: https://hf.co/pyannote/segmentation-3.0")
                print("   Click 'Accept' on the license agreement")
                
            elif "wespeaker" in str(e) or "embedding" in str(e):
                print("\n🎯 Missing license for speaker embedding model:")
                print("   Visit: https://hf.co/pyannote/wespeaker-voxceleb-resnet34-LM")
                print("   Click 'Accept' on the license agreement")
                
            elif "diarization" in str(e):
                print("\n🎯 Missing license for main diarization model:")
                print("   Visit: https://hf.co/pyannote/speaker-diarization-3.1")
                print("   Click 'Accept' on the license agreement")
                
            elif "token" in str(e).lower():
                print("\n🔑 Token issue:")
                print("   1. Check your token at: https://huggingface.co/settings/tokens")
                print("   2. Make sure it has 'Read' access")
                
            else:
                print(f"\n❓ Unexpected error. Run 'python check_licenses.py' for help.")
                
            print("\n⚠️  Remember: You need to accept licenses for ALL pyannote models!")
            return None