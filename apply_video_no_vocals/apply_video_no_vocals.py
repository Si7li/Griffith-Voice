import ffmpeg
import tempfile
import os

class VideoNoVocalsApplier:

    def __init__(self, final_translated_audio, no_vocals_path, input_video):
        self.final_translated_audio = final_translated_audio
        self.no_vocals_path = no_vocals_path
        self.input_video = input_video

    def mix_audios(self, output_path, voice_volume=1.0, background_volume=0.3, master_volume=1.2):
        """
        Mix final translated audio with background audio, preserving volumes.
        
        Args:
            output_path (str): Output path for mixed audio
            voice_volume (float): Volume multiplier for translated voice (1.0 = original)
            background_volume (float): Volume multiplier for background (0.3 = 30% of original)
            master_volume (float): Overall amplification after mixing (1.2 = 20% boost)
        """
        # Create volume-adjusted inputs
        voice_input = ffmpeg.input(self.final_translated_audio).filter('volume', voice_volume)
        background_input = ffmpeg.input(self.no_vocals_path).filter('volume', background_volume)
        
        # Mix the audio streams
        mixed = ffmpeg.filter(
            [voice_input, background_input],
            'amix', 
            inputs=2, 
            duration='longest',
            dropout_transition=0,
            normalize=0  # Prevent automatic volume reduction
        )
        
        # Apply master volume amplification to the mixed result
        amplified = mixed.filter('volume', master_volume)
        
        (
            amplified
            .output(output_path, acodec='pcm_s16le')  # Keep WAV quality
            .overwrite_output()
            .run()
        )
        return output_path
    
    def replace_video_audio(self, mixed_audio_path, output_video_path):
        """Replace video's audio with the mixed audio."""
        (
            ffmpeg
            .output(
                ffmpeg.input(self.input_video, an=None),  # keep video only
                ffmpeg.input(mixed_audio_path),          # add new audio
                output_video_path,
                vcodec='copy',  # don't re-encode video
                acodec='aac',   # compress audio for mp4
                strict='experimental'
            )
            .overwrite_output()
            .run()
        )
    
    def process(self, mixed_audio_out, final_video_out, voice_volume=1.0, background_volume=0.3, master_volume=1.2):
        """
        Full process: mix audios and replace video's audio.
        
        Args:
            mixed_audio_out (str): Output path for mixed audio
            final_video_out (str): Output path for final video
            voice_volume (float): Volume for translated voice (1.0 = original)
            background_volume (float): Volume for background music (0.3 = 30%)
            master_volume (float): Overall amplification after mixing (1.2 = 20% boost)
        """
        mixed_audio = self.mix_audios(mixed_audio_out, voice_volume, background_volume, master_volume)
        self.replace_video_audio(mixed_audio, final_video_out)