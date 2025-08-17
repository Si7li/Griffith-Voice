import whisper
import os
import glob
import torch
from collections import defaultdict
from utils import save_cache, read_cache

class AudioTranscriber:
    def __init__(self, model_size="small"):

        print(f"Loading Whisper model: {model_size}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)
        self.model_size = model_size
    
    def transcribe_folder(self, segments_folder, diarization_data=None, language=None, read_from_cache=False, cache_path=None):
        
        # Try loading from cache first
        transcriptions = read_cache(read_from_cache, cache_path)
        if transcriptions:
            print(f"Using cached transcriptions from: {cache_path}")
            return transcriptions
        
        if not os.path.exists(segments_folder):
            print(f"Error: Segments folder '{segments_folder}' not found")
            return None
        
        # Find all audio segment files
        audio_files = glob.glob(os.path.join(segments_folder, "*.wav"))
        
        if not audio_files:
            print(f"No .wav files found in {segments_folder}")
            return None
        
        print(f"Found {len(audio_files)} audio segments to transcribe")
        
        # Group files by speaker
        transcriptions = defaultdict(list)
        
        for audio_file in sorted(audio_files):
            filename = os.path.basename(audio_file)
            
            # Parse filename: SPEAKER_XX_segY.wav
            if not filename.startswith("SPEAKER_"):
                continue
                
            try:
                # Extract speaker and segment info
                parts = filename.replace(".wav", "").split("_")
                speaker_id = f"{parts[0]}_{parts[1]}"  # SPEAKER_00
                segment_num = int(parts[2].replace("seg", ""))
                
                print(f"Transcribing {filename}...")
                
                # Transcribe the audio
                result = self.model.transcribe(
                    audio_file, 
                    language=language,
                    verbose=False,
                    word_timestamps=False  # Disable word-level timestamps to avoid "word:" output
                )
                
                # Get timing info from diarization data if provided
                start_time = None
                end_time = None
                if diarization_data and speaker_id in diarization_data:
                    if segment_num < len(diarization_data[speaker_id]):
                        start_time, end_time = diarization_data[speaker_id][segment_num]
                
                # Calculate confidence from segments or use no_speech_prob
                confidence = None
                if "segments" in result and result["segments"]:
                    # Average confidence from all segments
                    segment_probs = []
                    for seg in result["segments"]:
                        if "avg_logprob" in seg:
                            # Convert log probability to regular probability (0-1)
                            prob = min(1.0, max(0.0, 1.0 + seg["avg_logprob"]))
                            segment_probs.append(prob)
                    if segment_probs:
                        confidence = sum(segment_probs) / len(segment_probs)
                elif "no_speech_prob" in result:
                    # Use inverse of no_speech_prob as confidence
                    confidence = 1.0 - result["no_speech_prob"]
                
                if confidence > 0.2:
                    print(confidence)
                    # Store transcription with metadata
                    transcription_data = {
                        "text": result["text"].strip(),
                        "language": result.get("language", "unknown"),
                        "file": filename,
                        "segment_num": segment_num,
                        "start": start_time,  # From diarization
                        "end": end_time,      # From diarization
                        "confidence": confidence
                    }
                    
                    transcriptions[speaker_id].append(transcription_data)
                    print(f"  → '{result['text'].strip()}'")  # Quote the text to see exact content
                
            except Exception as e:
                print(f"Error transcribing {filename}: {e}")
                continue
        
        # Sort segments by segment number within each speaker
        for speaker_id in transcriptions:
            transcriptions[speaker_id].sort(key=lambda x: x["segment_num"])
        
        # Save to cache
        if cache_path:
            save_cache(cache_path, dict(transcriptions))
            print(f"Transcriptions cached to: {cache_path}")
        
        # Unload Whisper model and free GPU memory
        del self.model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("🧹 Whisper model unloaded and GPU memory cleared.")
        
        print(f"✓ Transcription completed! {len(transcriptions)} speakers")
        return dict(transcriptions)
    
        