import os
import sys
import soundfile as sf
import numpy as np
from pathlib import Path

class TranslationsSynthensizer:
    def __init__(self):
        pass

# Add the GPT-SoVITS directory to Python path
gpt_sovits_path = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS"
sys.path.append(gpt_sovits_path)

# Change to the GPT-SoVITS directory before importing modules
original_cwd = os.getcwd()
os.chdir(gpt_sovits_path)

# Set the correct BERT and CNHubert paths before importing GPT-SoVITS modules
os.environ["bert_path"] = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
os.environ["cnhubert_base_path"] = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"

# Import GPT-SoVITS modules directly
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav

# Change back to original directory
os.chdir(original_cwd)

# Initialize i18n (do this from the GPT-SoVITS directory)
os.chdir(gpt_sovits_path)
i18n = I18nAuto()
os.chdir(original_cwd)

# Model paths - using the EXACT same models as the webui
# Based on webui terminal output, it's using v2 models initially, then v2ProPlus
# Let's match the webui exactly by using the v2 models that it loads first
gpt_model_path = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
sovits_model_path = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS/GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"

# Audio and text inputs
reference_wav = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/voice_samples/SPEAKER_00_voice_sample.wav"
reference_text = "そんな一生、男なら一度は思い描くはずです。夢という名の神の、順教者としての一生。"  # Japanese reference text
reference_language = "日文"  # Japanese

# Multiple reference audio files for the same character 
# These will be used to average the tone (like the webui does)
# Only the main reference_wav needs corresponding text
other_reference_wavs = [
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg0.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg1.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg2.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg3.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg4.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg5.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg6.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg7.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg8.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg9.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg10.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg11.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg12.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg13.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg14.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg15.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg16.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg17.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg18.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg19.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg20.wav", 
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg21.wav",
    "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/audio_segments/SPEAKER_00_seg22.wav"
]

# Target text to synthesize
target_text = "And if anyone tramples on that dream, they will confront them with their entire being."
target_language = "英文"  # English

# Output directory
output_dir = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/translated_outputs"
os.makedirs(output_dir, exist_ok=True)

def synthesize_with_multiple_references():
    """
    Synthesize audio using multiple reference files to improve voice cloning quality
    This mimics how the WebUI handles multiple reference files
    """
    print("Loading models...")
    
    # Load the models
    change_gpt_weights(gpt_path=gpt_model_path)
    change_sovits_weights(sovits_path=sovits_model_path)
    
    print("Models loaded successfully!")
    
    # Filter existing reference files
    existing_ref_files = []
    for ref_file in other_reference_wavs:
        if os.path.exists(ref_file):
            existing_ref_files.append(ref_file)
        else:
            print(f"Warning: Reference file {ref_file} not found, skipping...")
    
    print(f"Found {len(existing_ref_files)} additional reference files")
    
    # Create file objects that mimic what gradio would pass
    # Each file needs to have a .name attribute pointing to the file path
    class FileObject:
        def __init__(self, file_path):
            self.name = file_path
    
    inp_refs = [FileObject(ref_file) for ref_file in existing_ref_files] if existing_ref_files else None
    
    print(f"Generating audio using main reference + {len(existing_ref_files) if existing_ref_files else 0} additional references...")
    
    try:
        # Use the webui's get_tts_wav function with EXACT webui parameters
        synthesis_result = get_tts_wav(
            ref_wav_path=reference_wav,
            prompt_text=reference_text,
            prompt_language=i18n(reference_language),
            text=target_text,
            text_language=i18n(target_language),
            how_to_cut=i18n("不切"),  # webui default (don't cut)
            top_k=15,                # webui default
            top_p=0.5,                 # webui default  
            temperature=1,           # webui default
            ref_free=False,          # webui default
            speed=1,                 # webui default
            if_freeze=False,         # webui default
            inp_refs=inp_refs,       # multiple references
            sample_steps=8,          # webui default for v2ProPlus
            if_sr=False,             # webui default
            pause_second=0.3,        # webui default
        )
        
        result_list = list(synthesis_result)
        if result_list:
            sampling_rate, audio_data = result_list[-1]
            
            # Save the result
            output_wav_path = os.path.join(output_dir, "output.wav")
            sf.write(output_wav_path, audio_data, sampling_rate)
            print(f"Audio generated and saved to: {output_wav_path}")
            
            return output_wav_path
        else:
            print("Error: No audio was generated!")
            return None
            
    except Exception as e:
        print(f"Error during synthesis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        output_path = synthesize_with_multiple_references()
        print("Voice cloning completed successfully!")
        print(f"Output file: {output_path}")
    except Exception as e:
        print(f"Error during voice cloning: {str(e)}")
        import traceback
        traceback.print_exc()