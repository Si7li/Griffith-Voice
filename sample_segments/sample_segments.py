import os
import ffmpeg
from glob import glob
from collections import defaultdict
import librosa
import numpy as np
from pydub import AudioSegment
from utils import save_cache, read_cache

class SegmentsSampler:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.min_duration = 3.0  # minimum 3 seconds
        self.max_duration = 10.0  # maximum 10 seconds
        self.max_segments = 5  # maximum number of segments to combine

    def _get_audio_features(self, filepath):
        """Extract audio features for diversity analysis"""
        try:
            y, sr = librosa.load(filepath, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            
            if duration < 0.5:  # Skip very short segments
                return None
            
            # Extract features for diversity
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Calculate mean values for comparison
            features = {
                'duration': duration,
                'mfcc_mean': np.mean(mfccs),
                'mfcc_std': np.std(mfccs),
                'spectral_centroid': np.mean(spectral_centroid),
                'spectral_rolloff': np.mean(spectral_rolloff),
                'zcr': np.mean(zero_crossing_rate),
                'energy': np.mean(librosa.feature.rms(y=y))
            }
            return features
        except Exception as e:
            print(f"⚠️  Error processing {filepath}: {e}")
            return None

    def _select_diverse_segments(self, files, transcribed_data=None):
        """Select multiple diverse audio segments for best voice variety"""
        valid_segments = []
        
        # Filter segments by duration and extract features
        for filepath in files:
            features = self._get_audio_features(filepath)
            if features and features['duration'] >= 0.5:  # At least 0.5 seconds
                # Extract segment info for transcription lookup
                filename = os.path.basename(filepath)
                segment_info = self._parse_segment_filename(filename)
                
                segment_data = {
                    'path': filepath,
                    'features': features,
                    'filename': filename,
                    'speaker_id': segment_info['speaker_id'],
                    'segment_num': segment_info['segment_num']
                }
                
                # Add transcription and translation if available
                if transcribed_data and segment_info['speaker_id'] in transcribed_data:
                    speaker_segments = transcribed_data[segment_info['speaker_id']]
                    for seg in speaker_segments:
                        if seg.get('segment_num') == segment_info['segment_num']:
                            segment_data['transcription'] = seg.get('text', '')
                            segment_data['translation'] = seg.get('translation', '')
                            break
                
                valid_segments.append(segment_data)
        
        if not valid_segments:
            print(f"⚠️  No valid segments found")
            return []
        
        # Sort by quality factors: duration, energy, and diversity potential
        valid_segments.sort(key=lambda x: (
            x['features']['duration'],  # Prefer longer segments
            x['features']['energy'],    # Prefer higher energy (clearer voice)
            -x['features']['mfcc_std']  # Prefer more varied vocal characteristics
        ), reverse=True)
        
        selected = []
        total_duration = 0.0
        feature_vectors = []
        
        for segment in valid_segments:
            if len(selected) >= self.max_segments:
                break
                
            duration = segment['features']['duration']
            
            # Check if adding this segment would exceed max duration
            if total_duration + duration > self.max_duration:
                # Try to fit a shorter segment if we haven't reached min_duration
                if total_duration < self.min_duration:
                    continue
                else:
                    break
            
            # Create feature vector for diversity check
            feature_vector = np.array([
                segment['features']['mfcc_mean'],
                segment['features']['spectral_centroid'] / 1000,  # normalize
                segment['features']['spectral_rolloff'] / 1000,   # normalize
                segment['features']['zcr'] * 100,                 # scale up
                segment['features']['energy']
            ])
            
            # Check diversity - avoid too similar segments (more lenient for multiple segments)
            is_diverse = True
            if feature_vectors:  # Only check if we have existing segments
                for existing_vector in feature_vectors:
                    similarity = np.corrcoef(feature_vector, existing_vector)[0, 1]
                    if not np.isnan(similarity) and similarity > 0.6:  # More lenient for variety
                        is_diverse = False
                        break
            
            if is_diverse or len(selected) == 0:  # Always take the first segment
                selected.append(segment)
                feature_vectors.append(feature_vector)
                total_duration += duration
                
                # Continue looking for more segments until we reach optimal duration
                if total_duration >= self.max_duration:
                    break
        
        return selected

    def _parse_segment_filename(self, filename):
        """Parse segment filename to extract speaker ID and segment number"""
        # Expected format: SPEAKER_XX_segY.wav
        try:
            parts = filename.replace('.wav', '').split('_')
            speaker_id = f"{parts[0]}_{parts[1]}"  # SPEAKER_00
            segment_num = int(parts[2].replace('seg', ''))
            return {
                'speaker_id': speaker_id,
                'segment_num': segment_num
            }
        except:
            return {
                'speaker_id': 'UNKNOWN',
                'segment_num': -1
            }

    def _group_segments_per_speaker(self):
        os.makedirs(self.output_folder, exist_ok=True)
        speaker_segments = defaultdict(list)

        for filepath in sorted(glob(os.path.join(self.input_folder, "*.wav"))):
            filename = os.path.basename(filepath)
            if filename.startswith("SPEAKER_"):
                speaker_id = filename.split("_seg")[0]
                speaker_segments[speaker_id].append(filepath)

        return speaker_segments
    
    def merge(self, transcribed_data=None):
        speaker_segments = self._group_segments_per_speaker()
        merged_files = {}
        
        for speaker_id, files in speaker_segments.items():
            print(f"🔍 Processing {speaker_id} with {len(files)} segments...")
            
            # Select diverse segments for voice cloning with transcription data
            selected_segments = self._select_diverse_segments(files, transcribed_data)
            
            if not selected_segments:
                print(f"❌ No suitable segments found for {speaker_id}")
                continue
            
            print(f"✅ Selected {len(selected_segments)} diverse segments for {speaker_id}")
            
            # Create concatenated audio from selected segments
            combined_audio = AudioSegment.empty()
            total_duration = 0
            combined_transcription = []
            combined_translation = []
            
            for i, segment in enumerate(selected_segments):
                audio = AudioSegment.from_wav(segment['path'])
                combined_audio += audio
                total_duration += segment['features']['duration']
                
                # Collect transcription and translation
                transcription = segment.get('transcription', '')
                translation = segment.get('translation', '')
                
                print(f"   + {segment['filename']} ({segment['features']['duration']:.1f}s)")
                if transcription:
                    print(f"     📝 Text: {transcription}")
                if translation:
                    print(f"     🌐 Translation: {translation}")
                
                if transcription:
                    combined_transcription.append(transcription)
                if translation:
                    combined_translation.append(translation)
            
            # Trim to max duration if needed
            if total_duration > self.max_duration:
                max_ms = int(self.max_duration * 1000)
                combined_audio = combined_audio[:max_ms]
                total_duration = self.max_duration
            
            # Export the merged audio
            output_path = os.path.join(self.output_folder, f"{speaker_id}_voice_sample.wav")
            combined_audio.export(output_path, format="wav")
            
            # Save transcription and translation files
            if combined_transcription:
                transcription_path = os.path.join(self.output_folder, f"{speaker_id}_transcription.txt")
                with open(transcription_path, 'w', encoding='utf-8') as f:
                    f.write(' '.join(combined_transcription))
            
            if combined_translation:
                translation_path = os.path.join(self.output_folder, f"{speaker_id}_translation.txt")
                with open(translation_path, 'w', encoding='utf-8') as f:
                    f.write(' '.join(combined_translation))
            
            merged_files[speaker_id] = {
                'audio_path': output_path,
                'duration': total_duration,
                'segments_count': len(selected_segments),
                'transcription': ' '.join(combined_transcription) if combined_transcription else '',
                'translation': ' '.join(combined_translation) if combined_translation else '',
                'transcription_file': transcription_path if combined_transcription else None,
                'translation_file': translation_path if combined_translation else None
            }
            
            print(f"🎵 Created voice sample for {speaker_id} → {output_path} ({total_duration:.1f}s)")
            if combined_transcription:
                print(f"📝 Full transcription: {' '.join(combined_transcription)}")
            if combined_translation:
                print(f"🌐 Full translation: {' '.join(combined_translation)}")
            
        print(f"✓ Voice sampling completed! {len(merged_files)} speakers processed")
        return merged_files