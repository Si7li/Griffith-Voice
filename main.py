from extract_audio import ExtractAudio
from separate_audio import SeparateAudio
from diarize_audio import AudioDiarization
from extract_segments import SegmentExtractor
from transcribe_audio_segments import AudioTranscriber
from translate_segments import SegmentsTranslator

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
    transcribed_segments = audio_transcriber.transcribe_folder(segments_folder="outputs/audio_segments",diarization_data=diarization, language=None,read_from_cache=True,cache_path="caches/transcribed_segments.pkl")
    segments_translator = SegmentsTranslator()
    print(segments_translator.translate_segments(transcribed_segments=transcribed_segments, read_from_cache=True, cache_path="caches/translation.pkl"))

if __name__ == "__main__":
    main()