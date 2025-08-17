from extract_audio import ExtractAudio
from separate_audio import SeparateAudio
from diarize_audio import AudioDiarization
from extract_segments import SegmentExtractor
from transcribe_audio_segments import AudioTranscriber
from translate_segments import SegmentsTranslator
from sample_segments import SegmentsSampler
from synthensize_translations import TranslationsSynthensizer, force_cleanup_gpt_sovits
from assemble_translations import AudioAssembler
from apply_video_no_vocals import VideoNoVocalsApplier
from utils import comprehensive_final_cleanup
import os
import shutil

def clear_output_directories():
    """Clear all output directories before processing a new video"""
    output_dirs = [
        "outputs",
        "outputs/audio_segments", 
        "outputs/voice_samples",
        "outputs/translated_outputs"
    ]
    
    print("🧹 Clearing output directories...")
    
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
                print(f"  ✅ Cleared: {dir_path}")
            except Exception as e:
                print(f"  ⚠️ Could not clear {dir_path}: {e}")
        else:
            # Create directory if it doesn't exist
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  📁 Created: {dir_path}")
            except Exception as e:
                print(f"  ⚠️ Could not create {dir_path}: {e}")
    
    print("🧹 Output directories cleared!")

def main():
    # Clear output directories before starting
    clear_output_directories()
    
    print("Starting audio extraction...")
    # Initialize audio extractor
    audio_extractor = ExtractAudio("inputs/input_video.mp4")
    # Extract audio from video input
    result_audio = audio_extractor.extract_audio("outputs/output_audio.wav",read_from_cache=False, cache_path="caches/audio_extracted.pkl")
    print(f"Extraction result: {result_audio}")
    # Initialize audio separator
    audio_separator = SeparateAudio(result_audio)
    # Seperate vocal from non-vocals from extracted video
    result_audio_separated = audio_separator.separate_audio(read_from_cache=False, cache_path="caches/audio_separated.pkl")
    print(f"Extraction result seperated: {result_audio_separated}")
    vocals = result_audio_separated['vocals']
    no_vocals = result_audio_separated['music']
    # Initialize audio diarizer
    audio_diarizer = AudioDiarization(vocals)
    diarization = audio_diarizer.diarize_audio(read_from_cache=False, cache_path="caches/diarization.pkl",min_segment_duration= 0)
    # Initialize segments extractor - use the vocals audio file, not the video file
    segments_extractor = SegmentExtractor(vocals, diarization)
    # Extract segments
    extracted = segments_extractor.extract_segments("outputs/audio_segments/")
    # Initialize audio transcriber
    audio_transcriber = AudioTranscriber("small")
    # Transcribe audio segments
    transcribed_segments = audio_transcriber.transcribe_folder(segments_folder="outputs/audio_segments",diarization_data=diarization, language="en",read_from_cache=False,cache_path="caches/transcribed_segments.pkl")
    # Initialize audio translator
    segments_translator = SegmentsTranslator()
    # Translate audio segments
    translated_segments = segments_translator.translate_segments(transcribed_segments=transcribed_segments,diarization_essensials=diarization , source_lang="en", target_lang="ja", read_from_cache=False, cache_path="caches/translation.pkl")
    # Initialize segments sampler
    segments_sampler = SegmentsSampler("outputs/audio_segments", "outputs/voice_samples")
    # Get a sample per speaker for voice-cloning
    audio_samples = segments_sampler.merge(transcribed_data=translated_segments, read_from_cache=False, cache_path="caches/voice_samples.pkl")
    
    # Initialize translations synthesizer
    translations_synthesizer = TranslationsSynthensizer()
    # Synthensize translated texts
    synthesis_results = translations_synthesizer.synthesize_translations(
        transcribed_segments=transcribed_segments,
        translated_segments=translated_segments,
        voice_samples_dir="outputs/voice_samples",
        audio_segments_dir="outputs/audio_segments",
        top_k=15,
        top_p=0.7,
        temperature=1,
        speed=1.1,
        prompt_language="en",
        target_language="ja",
        read_from_cache=False,
        cache_path="caches/synthesis_results.pkl")
    
    # Explicitly delete synthesizer to ensure GPU cleanup
    del translations_synthesizer
    
    # Force cleanup of GPT-SoVITS models
    force_cleanup_gpt_sovits()
    
    # Initialize audio assembler
    audio_assembler = AudioAssembler("inputs/input_video.mp4")
    # Assemble all translated audio segments into final audio track (conversation only)
    final_audio = audio_assembler.assemble_audio(
        synthesis_results=synthesis_results,
        output_path="outputs/final_translated_audio.wav",
        read_from_cache=False,
        cache_path="caches/assembled_audio.pkl")
    
    # Initialize Video and No_vocals applier
    video_no_vocals_applier = VideoNoVocalsApplier(final_translated_audio=final_audio, no_vocals_path=no_vocals, input_video="inputs/input_video.mp4")
    video_no_vocals_applier.process(
        mixed_audio_out="outputs/mixed.wav",
        final_video_out="outputs/output.mp4", 
        voice_volume=1.0,      # Keep voice at original level
        background_volume=0.3, # Background at 30% to not overpower voice
        master_volume=1.2      # 20% overall boost for better audibility
    )
    
    # Final comprehensive cleanup to ensure all models are unloaded
    comprehensive_final_cleanup()
    print("🧹 Processing complete - all models unloaded")
    

if __name__ == "__main__":
    main()