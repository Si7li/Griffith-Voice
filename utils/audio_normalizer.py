import numpy as np
import soundfile as sf
from scipy import signal
import librosa

class AudioVolumeNormalizer:
    """
    Audio volume normalization utility to ensure consistent volume levels
    throughout the translation pipeline.
    """
    
    def __init__(self, target_lufs=-23.0, peak_limit=-3.0):
        """
        Initialize volume normalizer with broadcast standards.
        
        Args:
            target_lufs (float): Target LUFS (Loudness Units relative to Full Scale)
                                -23.0 is broadcast standard, -16.0 for streaming
            peak_limit (float): Maximum peak level in dB to prevent clipping
        """
        self.target_lufs = target_lufs
        self.peak_limit = peak_limit
    
    def calculate_lufs(self, audio, sample_rate):
        """
        Calculate LUFS (Loudness Units relative to Full Scale) of audio.
        Simplified implementation for mono/stereo audio.
        """
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)
        
        # Convert to dB power
        rms = np.sqrt(np.mean(audio**2, axis=0))
        rms = np.maximum(rms, 1e-10)  # Prevent log(0)
        lufs = 20 * np.log10(rms) - 0.691  # Simplified LUFS approximation
        
        return float(np.mean(lufs))
    
    def normalize_peak(self, audio, target_peak_db=-3.0):
        """
        Normalize audio to target peak level.
        
        Args:
            audio (np.ndarray): Input audio
            target_peak_db (float): Target peak level in dB
            
        Returns:
            np.ndarray: Peak-normalized audio
        """
        current_peak = np.max(np.abs(audio))
        if current_peak == 0:
            return audio
            
        target_linear = 10**(target_peak_db / 20.0)
        gain = target_linear / current_peak
        
        return audio * gain
    
    def normalize_rms(self, audio, target_rms_db=-20.0):
        """
        Normalize audio to target RMS level.
        
        Args:
            audio (np.ndarray): Input audio
            target_rms_db (float): Target RMS level in dB
            
        Returns:
            np.ndarray: RMS-normalized audio
        """
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms == 0:
            return audio
            
        target_linear = 10**(target_rms_db / 20.0)
        gain = target_linear / current_rms
        
        # Apply gain with peak limiting
        normalized = audio * gain
        peak = np.max(np.abs(normalized))
        if peak > 0.95:  # Prevent clipping
            normalized = normalized / peak * 0.95
            
        return normalized
    
    def smart_normalize(self, audio, sample_rate):
        """
        Smart normalization that combines RMS and peak limiting.
        
        Args:
            audio (np.ndarray): Input audio
            sample_rate (int): Sample rate
            
        Returns:
            np.ndarray: Normalized audio
        """
        if len(audio) == 0 or np.max(np.abs(audio)) == 0:
            return audio
        
        # Step 1: RMS normalization for consistent loudness
        normalized = self.normalize_rms(audio, target_rms_db=-18.0)
        
        # Step 2: Peak limiting to prevent clipping
        peak = np.max(np.abs(normalized))
        if peak > 0.9:
            normalized = normalized / peak * 0.9
        
        # Step 3: Light compression for consistency
        normalized = self.apply_light_compression(normalized)
        
        return normalized
    
    def apply_light_compression(self, audio, threshold=0.7, ratio=3.0, makeup_gain=1.1):
        """
        Apply light compression to reduce dynamic range.
        
        Args:
            audio (np.ndarray): Input audio
            threshold (float): Compression threshold (0-1)
            ratio (float): Compression ratio
            makeup_gain (float): Makeup gain after compression
            
        Returns:
            np.ndarray: Compressed audio
        """
        # Simple soft knee compression
        abs_audio = np.abs(audio)
        compressed = np.where(
            abs_audio > threshold,
            threshold + (abs_audio - threshold) / ratio,
            abs_audio
        )
        
        # Preserve sign and apply makeup gain
        compressed_audio = np.sign(audio) * compressed * makeup_gain
        
        # Final peak limiting
        peak = np.max(np.abs(compressed_audio))
        if peak > 0.95:
            compressed_audio = compressed_audio / peak * 0.95
            
        return compressed_audio
    
    def normalize_file(self, input_path, output_path=None):
        """
        Normalize an audio file and save the result.
        
        Args:
            input_path (str): Path to input audio file
            output_path (str): Path to output file (if None, overwrites input)
            
        Returns:
            str: Path to normalized file
        """
        if output_path is None:
            output_path = input_path
            
        try:
            # Load audio
            audio, sample_rate = sf.read(input_path)
            
            # Normalize
            normalized_audio = self.smart_normalize(audio, sample_rate)
            
            # Save
            sf.write(output_path, normalized_audio, sample_rate)
            
            return output_path
            
        except Exception as e:
            print(f"Error normalizing {input_path}: {e}")
            return input_path
    
    def get_audio_stats(self, audio, sample_rate):
        """
        Get audio statistics for debugging.
        
        Args:
            audio (np.ndarray): Audio data
            sample_rate (int): Sample rate
            
        Returns:
            dict: Audio statistics
        """
        if len(audio) == 0:
            return {"peak_db": -np.inf, "rms_db": -np.inf, "lufs": -np.inf}
            
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        
        peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
        rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
        lufs = self.calculate_lufs(audio, sample_rate)
        
        return {
            "peak_db": round(peak_db, 2),
            "rms_db": round(rms_db, 2), 
            "lufs": round(lufs, 2),
            "duration": len(audio) / sample_rate
        }
