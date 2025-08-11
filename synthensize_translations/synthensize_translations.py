import os
import sys
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import glob
from utils import save_cache, read_cache

class TranslationsSynthensizer:
    def __init__(self, gpt_model_path=None, sovits_model_path=None):
        self.gpt_sovits_path = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS"
        self.output_dir = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/translated_outputs"
        
        # Default model paths
        self.gpt_model_path = gpt_model_path or "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        self.sovits_model_path = sovits_model_path or "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
        
        # Initialize GPT-SoVITS
        self._setup_gpt_sovits()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _setup_gpt_sovits(self):
        """Setup GPT-SoVITS environment and imports"""
        # Add the GPT-SoVITS directory to Python path
        sys.path.append(self.gpt_sovits_path)

        # Change to the GPT-SoVITS directory before importing modules
        self.original_cwd = os.getcwd()
        os.chdir(self.gpt_sovits_path)

        # Set the correct BERT and CNHubert paths before importing GPT-SoVITS modules
        os.environ["bert_path"] = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        os.environ["cnhubert_base_path"] = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"

        # Import GPT-SoVITS modules directly
        from tools.i18n.i18n import I18nAuto
        from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

        # Store the imports for later use
        self.change_gpt_weights = change_gpt_weights
        self.change_sovits_weights = change_sovits_weights
        self.get_tts_wav = get_tts_wav

        # Initialize i18n (do this from the GPT-SoVITS directory)
        self.i18n = I18nAuto()
        
        # Change back to original directory
        os.chdir(self.original_cwd)
        
        # Load models once
        print("Loading GPT-SoVITS models...")
        self.change_gpt_weights(gpt_path=self.gpt_model_path)
        self.change_sovits_weights(sovits_path=self.sovits_model_path)
        print("Models loaded successfully!")
        
    def synthesize_translations(self, transcribed_segments, translated_segments, voice_samples_dir, audio_segments_dir, target_language="英文", read_from_cache=False, cache_path=None):
        """
        Synthesize translated audio for all speakers
        
        Args:
            transcribed_segments: Dict with speaker transcriptions
            translated_segments: Dict with speaker translations  
            voice_samples_dir: Directory containing voice samples
            audio_segments_dir: Directory containing audio segments
            target_language: Target language for synthesis
            read_from_cache: Whether to try loading from cache first
            cache_path: Path to cache file
            
        Returns:
            Dict with synthesis results including timing information
        """
        # Try loading from cache first
        synthesis_results = read_cache(read_from_cache, cache_path)
        if synthesis_results:
            print(f"Using cached synthesis results from: {cache_path}")
            return synthesis_results
        
        synthesis_results = {}
        
        for speaker_id in transcribed_segments.keys():
            print(f"\nProcessing speaker: {speaker_id}")
            
            # Get voice sample and transcription for this speaker
            voice_sample_result = self._get_voice_sample_data(speaker_id, voice_samples_dir)
            if not voice_sample_result:
                print(f"Skipping {speaker_id} - no voice sample found")
                continue
                
            reference_wav, reference_text = voice_sample_result
            
            # Get other reference audio files for this speaker
            other_references = self._get_other_references(speaker_id, audio_segments_dir)
            
            # Get translations for this speaker
            if speaker_id not in translated_segments:
                print(f"No translations found for {speaker_id}")
                continue
                
            speaker_translations = translated_segments[speaker_id]
            
            # Synthesize each translated segment
            speaker_results = self._synthesize_speaker_segments(
                speaker_id=speaker_id,
                reference_wav=reference_wav,
                reference_text=reference_text,
                other_references=other_references,
                translations=speaker_translations,
                target_language=target_language
            )
            
            synthesis_results[speaker_id] = speaker_results
            
        # Save to cache
        if cache_path:
            save_cache(cache_path, synthesis_results)
            print(f"Synthesis results cached to: {cache_path}")
            
        return synthesis_results
    
    def _get_voice_sample_data(self, speaker_id, voice_samples_dir):
        """Get the voice sample wav file and its transcription for a speaker"""
        # Look for voice sample file
        voice_sample_pattern = os.path.join(voice_samples_dir, f"{speaker_id}_voice_sample.wav")
        if not os.path.exists(voice_sample_pattern):
            print(f"Voice sample not found: {voice_sample_pattern}")
            return None
            
        # Look for transcription file
        transcription_pattern = os.path.join(voice_samples_dir, f"{speaker_id}_transcription.txt")
        if not os.path.exists(transcription_pattern):
            print(f"Transcription not found: {transcription_pattern}")
            return None
            
        # Read transcription
        with open(transcription_pattern, 'r', encoding='utf-8') as f:
            reference_text = f.read().strip()
            
        return voice_sample_pattern, reference_text
    
    def _get_other_references(self, speaker_id, audio_segments_dir):
        """Get other reference audio files for a speaker"""
        pattern = os.path.join(audio_segments_dir, f"{speaker_id}_seg*.wav")
        other_refs = glob.glob(pattern)
        
        # Filter out existing files
        existing_refs = [ref for ref in other_refs if os.path.exists(ref)]
        print(f"Found {len(existing_refs)} additional reference files for {speaker_id}")
        
        return existing_refs
    
    def _split_long_text_smartly(self, text, max_length=200):
        """Smart text splitting that preserves meaning and handles rejoining"""
        if len(text) <= max_length:
            return [{'text': text, 'is_split': False}]
        
        import re
        
        # Try to split by natural sentence boundaries first
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence exceeds limit
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence + "."
            
            if len(test_chunk) <= max_length:
                current_chunk = test_chunk
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'is_split': True,
                        'chunk_index': len(chunks),
                        'is_continuation': len(chunks) > 0
                    })
                
                # Start new chunk with current sentence
                current_chunk = sentence + "."
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk,
                'is_split': len(chunks) > 0,
                'chunk_index': len(chunks),
                'is_continuation': len(chunks) > 0
            })
        
        return chunks

    def _synthesize_text_chunks(self, speaker_id, segment_num, text_chunks, reference_wav, reference_text, inp_refs, target_language, speaker_output_dir, start_time, end_time, original_text):
        """Synthesize multiple text chunks and combine them"""
        chunk_audio_files = []
        combined_audio = None
        
        for chunk_info in text_chunks:
            chunk_text = chunk_info['text']
            chunk_index = chunk_info.get('chunk_index', 0)
            is_continuation = chunk_info.get('is_continuation', False)
            
            # Adjust synthesis parameters for chunks
            chunk_filename = f"{speaker_id}_translated_seg{segment_num}"
            if chunk_info['is_split']:
                chunk_filename += f"_chunk{chunk_index}"
            chunk_filename += ".wav"
            
            chunk_output_path = os.path.join(speaker_output_dir, chunk_filename)
            
            print(f"  Synthesizing chunk {chunk_index + 1}/{len(text_chunks)}: '{chunk_text[:40]}...'")
            
            try:
                # Adjust parameters for continuation chunks
                synthesis_params = {
                    'ref_wav_path': reference_wav,
                    'prompt_text': reference_text,
                    'prompt_language': self.i18n("日文"),
                    'text': chunk_text,
                    'text_language': self.i18n(target_language),
                    'how_to_cut': self.i18n("不切"),
                    'top_k': 15,
                    'top_p': 0.7,
                    'temperature': 1,
                    'ref_free': False,
                    'speed': 1.1,
                    'if_freeze': False,
                    'inp_refs': inp_refs,
                    'sample_steps': 8,
                    'if_sr': False,
                    'pause_second': 0.1 if is_continuation else 0.3,  # Shorter pause for continuations
                }
                
                synthesis_result = self.get_tts_wav(**synthesis_params)
                result_list = list(synthesis_result)
                
                if result_list:
                    sampling_rate, audio_data = result_list[-1]
                    
                    # Save individual chunk
                    sf.write(chunk_output_path, audio_data, sampling_rate)
                    chunk_audio_files.append(chunk_output_path)
                    
                    # Combine audio chunks
                    if combined_audio is None:
                        combined_audio = audio_data
                    else:
                        # Add small silence between chunks (0.1 seconds)
                        silence_samples = int(0.1 * sampling_rate)
                        silence = np.zeros(silence_samples, dtype=audio_data.dtype)
                        combined_audio = np.concatenate([combined_audio, silence, audio_data])
                    
                    print(f"    ✓ Chunk {chunk_index + 1} synthesized")
                    
                else:
                    print(f"    ✗ Failed to synthesize chunk {chunk_index + 1}")
                    return None
                    
            except Exception as e:
                print(f"    ✗ Error synthesizing chunk {chunk_index + 1}: {str(e)}")
                return None
        
        # Save combined audio file
        if combined_audio is not None:
            final_output_path = os.path.join(speaker_output_dir, f"{speaker_id}_translated_seg{segment_num}.wav")
            sf.write(final_output_path, combined_audio, sampling_rate)
            
            # Clean up individual chunk files (optional)
            for chunk_file in chunk_audio_files:
                try:
                    os.remove(chunk_file)
                except:
                    pass
            
            return {
                'segment_num': segment_num,
                'output_file': final_output_path,
                'translated_text': ' '.join([chunk['text'] for chunk in text_chunks]),
                'original_text': original_text,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time if (start_time is not None and end_time is not None) else None,
                'sampling_rate': sampling_rate,
                'audio_length_seconds': len(combined_audio) / sampling_rate,
                'was_split': len(text_chunks) > 1,
                'num_chunks': len(text_chunks)
            }
        
        return None

    def _synthesize_speaker_segments(self, speaker_id, reference_wav, reference_text, other_references, translations, target_language):
        """Synthesize all segments for a single speaker with smart text handling"""
        speaker_results = {
            'segments': [],
            'speaker_id': speaker_id,
            'reference_wav': reference_wav,
            'reference_text': reference_text
        }
        
        # Create speaker output directory
        speaker_output_dir = os.path.join(self.output_dir, speaker_id)
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        # Create file objects for other references
        class FileObject:
            def __init__(self, file_path):
                self.name = file_path
        
        inp_refs = [FileObject(ref_file) for ref_file in other_references] if other_references else None
        
        for segment in translations:
            segment_num = segment.get('segment_num', 0)
            translated_text = segment.get('translation', '')
            start_time = segment.get('start')
            end_time = segment.get('end')
            original_text = segment.get('text', '')
            
            if not translated_text.strip():
                print(f"Skipping empty translation for {speaker_id} segment {segment_num}")
                continue
            
            print(f"Synthesizing {speaker_id} segment {segment_num}: '{translated_text[:50]}...'")
            
            # Smart text splitting
            text_chunks = self._split_long_text_smartly(translated_text, max_length=180)
            
            if len(text_chunks) == 1 and not text_chunks[0]['is_split']:
                # Single chunk - use original method
                try:
                    synthesis_result = self.get_tts_wav(
                        ref_wav_path=reference_wav,
                        prompt_text=reference_text,
                        prompt_language=self.i18n("日文"),
                        text=translated_text,
                        text_language=self.i18n(target_language),
                        how_to_cut=self.i18n("不切"),
                        top_k=15,
                        top_p=0.7,
                        temperature=1,
                        ref_free=False,
                        speed=1.1,
                        if_freeze=False,
                        inp_refs=inp_refs,
                        sample_steps=8,
                        if_sr=False,
                        pause_second=0.3,
                    )
                    
                    result_list = list(synthesis_result)
                    if result_list:
                        sampling_rate, audio_data = result_list[-1]
                        
                        output_filename = f"{speaker_id}_translated_seg{segment_num}.wav"
                        output_wav_path = os.path.join(speaker_output_dir, output_filename)
                        sf.write(output_wav_path, audio_data, sampling_rate)
                        
                        segment_result = {
                            'segment_num': segment_num,
                            'output_file': output_wav_path,
                            'translated_text': translated_text,
                            'original_text': original_text,
                            'start_time': start_time,
                            'end_time': end_time,
                            'duration': end_time - start_time if (start_time is not None and end_time is not None) else None,
                            'sampling_rate': sampling_rate,
                            'audio_length_seconds': len(audio_data) / sampling_rate,
                            'was_split': False
                        }
                        
                        speaker_results['segments'].append(segment_result)
                        print(f"  ✓ Saved to: {output_wav_path}")
                    else:
                        print(f"  ✗ No audio generated for segment {segment_num}")
                        
                except Exception as e:
                    print(f"  ✗ Error synthesizing segment {segment_num}: {str(e)}")
                    continue
                
            else:
                # Multiple chunks - use chunk synthesis and combination
                print(f"  Text is long ({len(translated_text)} chars), splitting into {len(text_chunks)} chunks")
                
                segment_result = self._synthesize_text_chunks(
                    speaker_id, segment_num, text_chunks, reference_wav, reference_text,
                    inp_refs, target_language, speaker_output_dir, start_time, end_time, original_text
                )
                
                if segment_result:
                    speaker_results['segments'].append(segment_result)
                    print(f"  ✓ Combined audio saved to: {segment_result['output_file']}")
                else:
                    print(f"  ✗ Failed to synthesize long segment {segment_num}")
        
        # Sort segments by segment number
        speaker_results['segments'].sort(key=lambda x: x['segment_num'])
        
        # Save segment metadata
        metadata_file = os.path.join(speaker_output_dir, f"{speaker_id}_synthesis_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(speaker_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Completed {speaker_id}: {len(speaker_results['segments'])} segments synthesized")
        return speaker_results