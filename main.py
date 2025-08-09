from extract_audio import ExtractAudio
from separate_audio import SeparateAudio
from diarize_audio import AudioDiarization
from extract_segments import SegmentExtractor
from transcribe_audio_segments import AudioTranscriber
from translate_segments import SegmentsTranslator
from sample_segments import SegmentsSampler
from synthensize_translations import TranslationsSynthensizer
from assemble_translations import AudioAssembler
from apply_video_no_vocals import VideoNoVocalsApplier

def main():
    print("Starting audio extraction...")
    # Initialize audio extractor
    audio_extractor = ExtractAudio("inputs/input_video.mp4")
    # Extract audio from video input
    result_audio = audio_extractor.extract_audio("outputs/output_audio.wav",read_from_cache=True, cache_path="caches/audio_extracted.pkl")
    print(f"Extraction result: {result_audio}")
    # Initialize audio separator
    audio_separator = SeparateAudio(result_audio)
    # Seperate vocal from non-vocals from extracted video
    result_audio_separated = audio_separator.separate_audio(read_from_cache=True, cache_path="caches/audio_separated.pkl")
    print(f"Extraction result seperated: {result_audio_separated}")
    vocals = result_audio_separated['vocals']
    no_vocals = result_audio_separated['music']
    # Initialize audio diarizer
    audio_diarizer = AudioDiarization(vocals)
    diarization = audio_diarizer.diarize_audio(read_from_cache=True, cache_path="caches/diarization.pkl")
    # Initialize segments extractor - use the vocals audio file, not the video file
    segments_extractor = SegmentExtractor(vocals, diarization)
    # Extract segments
    extracted = segments_extractor.extract_segments("outputs/audio_segments/")
    # Initialize audio transcriber
    audio_transcriber = AudioTranscriber("small")
    # Transcribe audio segments
    transcribed_segments = audio_transcriber.transcribe_folder(segments_folder="outputs/audio_segments",diarization_data=diarization, language=None,read_from_cache=True,cache_path="caches/transcribed_segments.pkl")
    # Initialize audio translator
    segments_translator = SegmentsTranslator()
    # Translate audio segments
    translated_segments = segments_translator.translate_segments(transcribed_segments=transcribed_segments, read_from_cache=True, cache_path="caches/translation.pkl")
    # Initialize segments sampler
    segments_sampler = SegmentsSampler("outputs/audio_segments", "outputs/voice_samples")
    # Get a sample per speaker for voice-cloning
    audio_samples = segments_sampler.merge(transcribed_data=translated_segments, read_from_cache=True, cache_path="caches/voice_samples.pkl")
    
    # Initialize translations synthesizer
    translations_synthesizer = TranslationsSynthensizer()
    # Synthensize translated texts
    synthesis_results = translations_synthesizer.synthesize_translations(
        transcribed_segments=transcribed_segments,
        translated_segments=translated_segments,
        voice_samples_dir="outputs/voice_samples",
        audio_segments_dir="outputs/audio_segments",
        target_language="英文",  # English
        read_from_cache=True,
        cache_path="caches/synthesis_results.pkl")
    
    # Initialize audio assembler
    audio_assembler = AudioAssembler("inputs/input_video.mp4")
    # Assemble all translated audio segments into final audio track (conversation only)
    final_audio = audio_assembler.assemble_audio(
        synthesis_results=synthesis_results,
        output_path="outputs/final_translated_audio.wav",
        read_from_cache=True,
        cache_path="caches/assembled_audio.pkl")
    
    # Initialize Video and No_vocals applier
    video_no_vocals_applier = VideoNoVocalsApplier(final_translated_audio=final_audio, no_vocals_path=no_vocals, input_video="inputs/input_video.mp4")
    video_no_vocals_applier.process(mixed_audio_out="outputs/mixed.wav",final_video_out= "outputs/output.mp4")
    
    
if __name__ == "__main__":
    main()