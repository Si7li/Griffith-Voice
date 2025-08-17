import os
import ffmpeg
import tempfile
from utils import save_cache, read_cache
from utils.audio_normalizer import AudioVolumeNormalizer


class AudioAssembler:
    def __init__(self, video_input_path):
        """
        Initialize AudioAssembler with the original video path to get duration and audio properties.
        
        Args:
            video_input_path (str): Path to the original video file
        """
        self.video_input_path = video_input_path
        self.video_duration = self._get_video_duration()
        
        # Initialize audio normalizer for consistent volume
        self.audio_normalizer = AudioVolumeNormalizer(target_lufs=-18.0, peak_limit=-3.0)
        
    def _get_video_duration(self):
        """Get the duration of the original video file."""
        try:
            probe = ffmpeg.probe(self.video_input_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            print(f"Error getting video duration: {e}")
            return None
    
    def assemble_audio(self, synthesis_results, output_path, read_from_cache=False, cache_path=None):
        """
        Assemble translated audio segments into a complete audio track matching the original video duration.
        
        Args:
            synthesis_results (dict): Dictionary containing translated audio segments for each speaker
            output_path (str): Path where the final assembled audio will be saved
            read_from_cache (bool): Whether to read from cache
            cache_path (str): Cache file path
            
        Returns:
            str: Path to the assembled audio file
        """
        # Check cache first
        cached_result = read_cache(read_from_cache, cache_path)
        if cached_result:
            return cached_result
            
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Collect all segments from all speakers
            all_segments = []
            for speaker_id, speaker_data in synthesis_results.items():
                for segment in speaker_data['segments']:
                    all_segments.append({
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'audio_file': segment['output_file'],
                        'speaker': speaker_id
                    })
            
            # Sort segments by start time
            all_segments.sort(key=lambda x: x['start_time'])
            
            # Create the final audio assembly
            final_audio = self._create_assembled_audio(all_segments, output_path)
            
            # Save to cache
            if cache_path:
                save_cache(cache_path, output_path)
                
            return output_path
            
        except Exception as e:
            print(f"Error assembling audio: {e}")
            return None
    
    def _create_assembled_audio(self, segments, output_path):
        """
        Create the final assembled audio using ffmpeg - conversation only.
        
        Args:
            segments (list): List of audio segments with timing information
            output_path (str): Output file path
        """
        try:
            # Create a temporary directory for intermediate files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Step 1: Create silence for the full video duration
                silence_path = os.path.join(temp_dir, "silence.wav")
                (
                    ffmpeg
                    .input('anullsrc=channel_layout=stereo:sample_rate=44100', f='lavfi', t=self.video_duration)
                    .output(silence_path)
                    .overwrite_output()
                    .run(quiet=True)
                )
                
                # Step 2: Create individual positioned segments
                segment_files = []
                for i, segment in enumerate(segments):
                    if not os.path.exists(segment['audio_file']):
                        print(f"Warning: Segment file not found: {segment['audio_file']}")
                        continue
                        
                    segment_output = os.path.join(temp_dir, f"segment_{i}.wav")
                    
                    # Position the segment at the correct time with silence padding
                    start_time = segment['start_time']
                    
                    # Use original segment audio without volume modification
                    input_audio = ffmpeg.input(segment['audio_file'])
                    
                    if start_time > 0:
                        # Add silence before the segment
                        silence_before = ffmpeg.input('anullsrc=channel_layout=stereo:sample_rate=44100', 
                                                    f='lavfi', t=start_time)
                        positioned = ffmpeg.concat(silence_before, input_audio, v=0, a=1)
                    else:
                        positioned = input_audio
                    
                    # Extend with silence to match video duration
                    (
                        positioned
                        .output(segment_output, t=self.video_duration)
                        .overwrite_output()
                        .run(quiet=True)
                    )
                    
                    segment_files.append(segment_output)
                
                # Step 3: Mix all conversation segments together
                if segment_files:
                    if len(segment_files) == 1:
                        # If only one segment file, no mixing needed - preserve original volume
                        (
                            ffmpeg
                            .input(segment_files[0])
                            .output(output_path, acodec='pcm_s16le', ar=44100)
                            .overwrite_output()
                            .run(quiet=True)
                        )
                    else:
                        # Multiple segments - use amix with normalize=0 to preserve volume
                        inputs = []
                        for segment_file in segment_files:
                            inputs.append(ffmpeg.input(segment_file))
                        
                        # Mix without automatic volume reduction
                        mixed = ffmpeg.filter(inputs, 'amix', inputs=len(inputs), duration='longest', normalize=0)
                        
                        # Output the final mixed audio
                        (
                            mixed
                            .output(output_path, acodec='pcm_s16le', ar=44100)
                            .overwrite_output()
                            .run(quiet=True)
                        )
                else:
                    # If no segments, just create silence
                    (
                        ffmpeg
                        .input(silence_path)
                        .output(output_path, acodec='pcm_s16le', ar=44100)
                        .overwrite_output()
                        .run(quiet=True)
                    )
                
                print(f"✓ Audio assembled successfully: {output_path}")
                
                # Apply final normalization to the assembled audio
                print("Applying final volume normalization...")
                normalized_path = self.audio_normalizer.normalize_file(output_path)
                print(f"✓ Audio normalized: {normalized_path}")
                
                return normalized_path
                
        except ffmpeg.Error as e:
            print(f"FFmpeg error during audio assembly: {e}")
            return None
        except Exception as e:
            print(f"Error during audio assembly: {e}")
            return None