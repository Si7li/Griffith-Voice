import os
import ffmpeg
from glob import glob
from collections import defaultdict
import librosa
import numpy as np
import torch
from pydub import AudioSegment
from utils import save_cache, read_cache
from transcribe_audio_segments import AudioTranscriber

class SegmentsSampler:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.min_duration = 3.0  # minimum 3 seconds
        self.max_duration = 10.0  # maximum 10 seconds
        self.max_segments = 5  # maximum number of segments to combine
        self.transcriber = AudioTranscriber()  # Initialize AudioTranscriber

    def _get_audio_features(self, filepath):
        """Extract audio features for quality analysis and smart segment selection"""
        try:
            # Use pydub for simpler, more reliable audio analysis
            audio = AudioSegment.from_wav(filepath)
            duration = len(audio) / 1000.0  # Convert to seconds
            
            if duration < 0.5:  # Skip very short segments
                return None
            
            # Simple but effective quality metrics
            volume_db = audio.dBFS  # Volume level
            
            # Calculate quality score based on duration and volume
            quality_score = 0
            
            # Duration scoring - prefer segments that can contribute to target range
            if 1.0 <= duration <= 4.0:  # Good individual segment length
                quality_score += 100
            elif 0.5 <= duration < 1.0:  # Short but usable
                quality_score += 70
            elif 4.0 < duration <= 8.0:  # Longer segments (good for single use)
                quality_score += 80
            else:
                quality_score += 30  # Very short or very long
            
            # Volume scoring - prefer clear, audible speech
            if -25 <= volume_db <= -5:  # Good volume range
                quality_score += 100
            elif -35 <= volume_db < -25:  # Decent volume
                quality_score += 70
            elif -45 <= volume_db < -35:  # Low but usable
                quality_score += 40
            else:  # Too quiet or too loud
                quality_score += 10
            
            # Simple spectral analysis for voice quality
            try:
                y, sr = librosa.load(filepath, sr=22050)
                
                # Check for silence or noise
                rms_energy = np.mean(librosa.feature.rms(y=y))
                if rms_energy > 0.01:  # Has decent energy
                    quality_score += 50
                
                # Check spectral characteristics for voice-like content
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                if 500 <= spectral_centroid <= 4000:  # Voice frequency range
                    quality_score += 30
                    
            except Exception:
                # If librosa fails, just use basic metrics
                pass
            
            features = {
                'duration': duration,
                'volume_db': volume_db,
                'quality_score': quality_score,
                'is_good_length': 1.0 <= duration <= 4.0,
                'is_good_volume': -35 <= volume_db <= -5
            }
            return features
        except Exception as e:
            print(f"⚠️  Error processing {filepath}: {e}")
            return None

    def _select_diverse_segments(self, files, transcribed_data=None):
        """Select and combine segments to create optimal 3-10 second voice references with maximum variety"""
        valid_segments = []
        
        # Filter segments and extract features
        for filepath in files:
            features = self._get_audio_features(filepath)
            if features and features['duration'] >= 0.5:
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
        
        # Sort by quality score (best first)
        valid_segments.sort(key=lambda x: x['features']['quality_score'], reverse=True)
        
        # NEW STRATEGY: Always prioritize VARIETY and MULTIPLE SEGMENTS
        # Strategy 1: Try to combine multiple diverse segments (PREFERRED)
        best_combination = []
        best_score = 0
        best_duration = 0
        
        print(f"   🎯 Prioritizing variety - trying to combine multiple segments...")
        
        # Try different combinations starting with highest quality segments
        for start_idx in range(min(8, len(valid_segments))):  # Check more combinations for variety
            current_combination = []
            current_duration = 0
            current_score = 0
            used_indices = set()
            
            # Start with this segment
            start_segment = valid_segments[start_idx]
            current_combination.append(start_segment)
            current_duration += start_segment['features']['duration']
            current_score += start_segment['features']['quality_score']
            used_indices.add(start_idx)
            
            # Try to add MORE segments for variety (aim for 3-5 segments)
            for segment_idx, segment in enumerate(valid_segments):
                if segment_idx in used_indices:
                    continue
                    
                potential_duration = current_duration + segment['features']['duration'] + 0.2  # Include gap time
                
                # Be more aggressive about adding segments for variety
                if potential_duration <= self.max_duration and len(current_combination) < 5:
                    current_combination.append(segment)
                    current_duration = potential_duration
                    current_score += segment['features']['quality_score']
                    used_indices.add(segment_idx)
                    
                    # Continue adding until we hit limits
                    if current_duration >= 8.0:  # Don't go too long
                        break
            
            # Evaluate this combination - HEAVILY favor multiple segments
            if current_duration >= self.min_duration:
                # BIG BONUS for variety (multiple segments)
                if len(current_combination) >= 4:
                    current_score += 200  # Huge bonus for 4+ segments
                elif len(current_combination) >= 3:
                    current_score += 150  # Big bonus for 3+ segments  
                elif len(current_combination) >= 2:
                    current_score += 100  # Good bonus for 2+ segments
                else:
                    current_score -= 50   # Penalty for single segment
                
                # Bonus for being in optimal range
                if 4.0 <= current_duration <= 8.0:
                    current_score += 100
                elif 3.0 <= current_duration <= 9.0:
                    current_score += 50
                
                # Extra bonus for more diverse content
                if len(current_combination) >= 3:
                    current_score += 75  # Variety bonus
                
                # Check if this is better than our current best
                if current_score > best_score or (current_score == best_score and len(current_combination) > len(best_combination)):
                    best_combination = current_combination
                    best_score = current_score
                    best_duration = current_duration
        
        # Strategy 2: Only use single segment if we absolutely can't combine (FALLBACK)
        if not best_combination or len(best_combination) == 1:
            print(f"   ⚠️  Couldn't find good combination, trying to force multiple segments...")
            
            # Force multiple segments by being more lenient with quality
            forced_combination = []
            total_dur = 0
            
            # Take top segments and force them together
            for segment in valid_segments[:6]:  # Try more segments
                if total_dur + segment['features']['duration'] + 0.2 <= self.max_duration:
                    forced_combination.append(segment)
                    total_dur += segment['features']['duration'] + 0.2
                    
                    # Stop when we have enough variety and duration
                    if len(forced_combination) >= 3 and total_dur >= self.min_duration:
                        break
            
            if len(forced_combination) >= 2 and total_dur >= self.min_duration:
                print(f"   🎯 Forced combination of {len(forced_combination)} segments: {total_dur:.1f}s")
                return forced_combination
        
        # Strategy 3: Single segment only as absolute last resort
        if not best_combination:
            # Only if we have a really good single segment and can't combine anything
            best_segment = valid_segments[0]
            duration = best_segment['features']['duration']
            
            if duration >= self.min_duration and best_segment['features']['quality_score'] > 200:
                print(f"   ⚠️  LAST RESORT: Using single high-quality segment: {best_segment['filename']} ({duration:.1f}s)")
                return [best_segment]
            
            # Even last resort: combine any segments we can
            fallback_combo = []
            fallback_dur = 0
            for seg in valid_segments[:4]:
                if fallback_dur + seg['features']['duration'] <= self.max_duration:
                    fallback_combo.append(seg)
                    fallback_dur += seg['features']['duration']
                    if fallback_dur >= self.min_duration and len(fallback_combo) >= 2:
                        break
            
            if fallback_combo and len(fallback_combo) >= 2:
                print(f"   🎯 Fallback combination: {len(fallback_combo)} segments")
                return fallback_combo
        
        if best_combination:
            segment_files = [s['filename'] for s in best_combination]
            print(f"   🎯 Selected diverse combination: {segment_files} (total: {best_duration:.1f}s)")
            return best_combination
        
        # Ultimate fallback
        if valid_segments:
            print(f"   ⚠️  Ultimate fallback: {valid_segments[0]['filename']}")
            return [valid_segments[0]]
        
        return []

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
    
    def merge(self, transcribed_data=None, read_from_cache=False, cache_path=None):
        # Try loading from cache first
        merged_files = read_cache(read_from_cache, cache_path)
        if merged_files:
            print(f"Using cached voice samples from: {cache_path}")
            return merged_files
        
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
            
            # Create concatenated audio from selected segments with gaps for variety
            combined_audio = AudioSegment.empty()
            total_duration = 0
            combined_transcription = []
            combined_translation = []
            segment_details = []
            
            for i, segment in enumerate(selected_segments):
                audio = AudioSegment.from_wav(segment['path'])
                combined_audio += audio
                total_duration += segment['features']['duration']
                
                # Add small gap between segments (200ms) for natural flow and variety
                if i < len(selected_segments) - 1:
                    gap = AudioSegment.silent(duration=200)  # 200ms gap
                    combined_audio += gap
                    total_duration += 0.2  # Add gap time
                
                # Collect transcription and translation with segment info
                transcription = segment.get('transcription', '')
                translation = segment.get('translation', '')
                segment_num = segment.get('segment_num', i+1)
                
                print(f"   + {segment['filename']} ({segment['features']['duration']:.1f}s) [Quality: {segment['features']['quality_score']:.0f}]")
                if transcription:
                    print(f"     📝 Text: {transcription}")
                    combined_transcription.append(f"[Segment {segment_num}] {transcription}")
                else:
                    combined_transcription.append(f"[Segment {segment_num}] [No transcription]")
                
                if translation:
                    print(f"     🌐 Translation: {translation}")
                    combined_translation.append(f"[Segment {segment_num}] {translation}")
                else:
                    combined_translation.append(f"[Segment {segment_num}] [No translation]")
                
                # Store segment details for organized output
                segment_details.append({
                    'segment_num': segment_num,
                    'filename': segment['filename'],
                    'duration': segment['features']['duration'],
                    'transcription': transcription,
                    'translation': translation,
                    'quality_score': segment['features']['quality_score']
                })
            
            # Ensure we're within the max duration limit
            if total_duration > self.max_duration:
                max_ms = int(self.max_duration * 1000)
                combined_audio = combined_audio[:max_ms]
                total_duration = self.max_duration
                print(f"   ✂️  Trimmed to {self.max_duration}s to stay within limits")
            
            # Export the merged audio
            output_path = os.path.join(self.output_folder, f"{speaker_id}_voice_sample.wav")
            combined_audio.export(output_path, format="wav")
            
            # Always save transcription and translation files with simple clean text
            transcription_path = os.path.join(self.output_folder, f"{speaker_id}_transcription.txt")
            translation_path = os.path.join(self.output_folder, f"{speaker_id}_translation.txt")
            
            # Simple transcription using Whisper directly
            print(f"🎙️  Transcribing voice sample...")
            try:
                # Use the transcriber's Whisper model directly
                result = self.transcriber.model.transcribe(output_path)
                transcribed_text = result.get('text', '').strip()
                
                # Write simple text to file
                with open(transcription_path, 'w', encoding='utf-8') as f:
                    f.write(transcribed_text)
                        
            except Exception as e:
                print(f"⚠️  Transcription error: {e}")
                with open(transcription_path, 'w', encoding='utf-8') as f:
                    f.write("Transcription failed")
            
            # Write simple translation file (just the combined translation)
            with open(translation_path, 'w', encoding='utf-8') as f:
                if segment_details and any(seg['translation'] for seg in segment_details):
                    # Simple combined translation text
                    continuous_translation = " ".join([seg['translation'] for seg in segment_details if seg['translation']])
                    f.write(continuous_translation)
                else:
                    f.write(f"[No translation available for {speaker_id}]")
            
            merged_files[speaker_id] = {
                'audio_path': output_path,
                'duration': total_duration,
                'segments_count': len(selected_segments),
                'transcription': " ".join([seg['transcription'] for seg in segment_details if seg['transcription']]),
                'translation': " ".join([seg['translation'] for seg in segment_details if seg['translation']]),
                'transcription_file': transcription_path,
                'translation_file': translation_path,
                'segment_details': segment_details
            }
            
            print(f"🎵 Created diverse voice sample for {speaker_id} → {output_path} ({total_duration:.1f}s)")
            print(f"📝 Transcription file: {transcription_path}")
            print(f"🌐 Translation file: {translation_path}")
            print(f"🎯 Variety achieved: {len(selected_segments)} segments combined")
            
            # Show preview of the simple combined content (like the old version)
            if len(segment_details) > 0:
                all_text = " ".join([seg['transcription'] for seg in segment_details if seg['transcription']])
                all_trans = " ".join([seg['translation'] for seg in segment_details if seg['translation']])
                if all_text:
                    print(f"📄 Full text: {all_text[:100]}{'...' if len(all_text) > 100 else ''}")
                if all_trans:
                    print(f"🔄 Full translation: {all_trans[:100]}{'...' if len(all_trans) > 100 else ''}")
            
        # Save to cache
        if cache_path:
            save_cache(cache_path, merged_files)
            print(f"Voice samples cached to: {cache_path}")
        
        # Unload Whisper model and free GPU memory
        del self.transcriber.model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("🧹 Whisper model unloaded and GPU memory cleared.")
            
        print(f"✓ Voice sampling completed! {len(merged_files)} speakers processed")
        return merged_files