from pydub import AudioSegment
import os

class SegmentExtractor:
    def __init__(self,audio_path ,diarization):
        self.diarization = diarization
        self.audio_path = audio_path
    def extract_segments(self, output_dir):
        audio = AudioSegment.from_wav(self.audio_path)
        os.makedirs(output_dir, exist_ok=True)
        extracted = {}
        
        for speaker, segments in self.diarization.items():
            extracted[speaker] = []
            for i, (start, end) in enumerate(segments):
                # Extract the audio segment
                segment = audio[start * 1000:end * 1000]  # Convert to milliseconds
                seg_path = os.path.join(output_dir, f"{speaker}_seg{i}.wav")
                
                # Export the segment to file
                segment.export(seg_path, format="wav")
                print(f"Exported {speaker} segment {i}: {start:.2f}s - {end:.2f}s -> {seg_path}")
                
                extracted[speaker].append((seg_path, start, end))
                
        print(f"✓ Extraction completed! {len(extracted)} speakers, total segments: {sum(len(segs) for segs in extracted.values())}")
        return extracted