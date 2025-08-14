import os
import sys
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import glob
from utils import save_cache, read_cache, cleanup_gpu_memory

def force_cleanup_gpt_sovits():
    """Force cleanup of GPT-SoVITS models - aggressive cleanup for maximum memory savings"""
    try:
        import gc
        import torch
        import sys
        
        # Try to access the GPT-SoVITS module if it's been imported
        gpt_sovits_path = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS"
        
        if gpt_sovits_path in sys.path:
            try:
                import GPT_SoVITS.inference_webui as inference_webui
                
                # Clear ALL models for maximum memory savings
                all_model_vars = [
                    'vq_model', 't2s_model', 'hifigan_model', 'bigvgan_model',
                    'bert_model', 'ssl_model'
                ]
                
                # Preserve only essential lightweight configuration:
                # 'hps', 'config', 'dict_language', 'tokenizer' - small config objects
                
                for var_name in all_model_vars:
                    if hasattr(inference_webui, var_name):
                        try:
                            model = getattr(inference_webui, var_name)
                            if model is not None:
                                if hasattr(model, 'cpu'):
                                    model = model.cpu()
                                if hasattr(model, 'to'):
                                    model = model.to('cpu')
                                del model
                            setattr(inference_webui, var_name, None)
                            print(f"🧹 Force cleared {var_name}")
                        except Exception as e:
                            print(f"Warning force clearing {var_name}: {e}")
                            try:
                                setattr(inference_webui, var_name, None)
                            except:
                                pass
                                
            except Exception as e:
                print(f"Could not access GPT-SoVITS inference module: {e}")
        
        # Aggressive garbage collection
        for _ in range(5):
            gc.collect()
        
        # Aggressive GPU cleanup
        if torch.cuda.is_available():
            for _ in range(5):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        print("🧹 Aggressive force cleanup of GPT-SoVITS completed")
        
    except Exception as e:
        print(f"Force cleanup failed: {e}")

class TranslationsSynthensizer:
    def __init__(self, gpt_model_path=None, sovits_model_path=None):
        self.gpt_sovits_path = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS"
        self.output_dir = "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/outputs/translated_outputs"
        
        # Default model paths (relative to GPT-SoVITS directory)
        self.gpt_model_path = gpt_model_path or "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        self.sovits_model_path = sovits_model_path or "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
        
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
        
        print("🔄 Importing GPT-SoVITS inference functions...")
        try:
            from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
            print("✅ Successfully imported inference functions")
        except Exception as e:
            print(f"❌ Failed to import inference functions: {e}")
            raise

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
        print(f"🔄 GPT model path: {self.gpt_model_path}")
        print(f"🔄 SoVITS model path: {self.sovits_model_path}")
        
        # Change back to GPT-SoVITS directory for loading
        os.chdir(self.gpt_sovits_path)
        try:
            print(f"🔄 Loading GPT model: {self.gpt_model_path}")
            gpt_result = self.change_gpt_weights(gpt_path=self.gpt_model_path)
            print(f"🔄 Initial GPT load result: {type(gpt_result)}")
            
            print(f"🔄 Loading SoVITS model: {self.sovits_model_path}")
            # SoVITS function is a generator, consume it properly
            sovits_generator = self.change_sovits_weights(
                sovits_path=self.sovits_model_path,
                prompt_language="中文",
                text_language="中文"
            )
            sovits_results = []
            try:
                for result in sovits_generator:
                    sovits_results.append(result)
            except Exception as e:
                print(f"🔄 SoVITS generator completed: {e}")
            
            print("Models loaded successfully!")
        finally:
            os.chdir(self.original_cwd)
    
    def _verify_model_files_exist(self):
        """Verify that the model files actually exist on disk"""
        import os
        
        print("🔍 Verifying model files exist...")
        
        # Convert relative paths to absolute paths from GPT-SoVITS directory
        gpt_full_path = os.path.join(self.gpt_sovits_path, self.gpt_model_path)
        sovits_full_path = os.path.join(self.gpt_sovits_path, self.sovits_model_path)
        
        if not os.path.exists(gpt_full_path):
            print(f"❌ GPT model file not found: {gpt_full_path}")
            return False
        else:
            print(f"✅ GPT model file exists: {gpt_full_path}")
            
        if not os.path.exists(sovits_full_path):
            print(f"❌ SoVITS model file not found: {sovits_full_path}")
            return False
        else:
            print(f"✅ SoVITS model file exists: {sovits_full_path}")
            
        return True
    
    def _manually_load_bert_ssl_models(self):
        """Manually load BERT and SSL models if they're missing"""
        try:
            import GPT_SoVITS.inference_webui as inference_webui
            
            # Check if models need to be loaded
            need_bert = not (hasattr(inference_webui, 'bert_model') and inference_webui.bert_model is not None)
            need_ssl = not (hasattr(inference_webui, 'ssl_model') and inference_webui.ssl_model is not None)
            
            if need_bert or need_ssl:
                print(f"🔄 Manually loading missing models: BERT={need_bert}, SSL={need_ssl}")
                
                # Change to GPT-SoVITS directory for loading
                original_cwd = os.getcwd()
                os.chdir(self.gpt_sovits_path)
                
                try:
                    # Get the necessary imports and settings
                    from transformers import AutoTokenizer, AutoModelForMaskedLM
                    from feature_extractor import cnhubert
                    import torch
                    
                    # Get device and half precision settings
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    is_half = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6
                    
                    # Load BERT model if needed
                    if need_bert:
                        bert_path = os.environ.get("bert_path", "/home/khalils/Desktop/Projects/Real-time_Voice_Translation/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
                        print(f"🔄 Loading BERT model from: {bert_path}")
                        
                        if not hasattr(inference_webui, 'tokenizer') or inference_webui.tokenizer is None:
                            inference_webui.tokenizer = AutoTokenizer.from_pretrained(bert_path)
                            print("✅ Tokenizer loaded")
                        
                        inference_webui.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
                        if is_half:
                            inference_webui.bert_model = inference_webui.bert_model.half().to(device)
                        else:
                            inference_webui.bert_model = inference_webui.bert_model.to(device)
                        print("✅ BERT model loaded")
                    
                    # Load SSL model if needed
                    if need_ssl:
                        print("🔄 Loading SSL model...")
                        inference_webui.ssl_model = cnhubert.get_model()
                        if is_half:
                            inference_webui.ssl_model = inference_webui.ssl_model.half().to(device)
                        else:
                            inference_webui.ssl_model = inference_webui.ssl_model.to(device)
                        print("✅ SSL model loaded")
                        
                    return True
                    
                except Exception as e:
                    print(f"❌ Error manually loading models: {e}")
                    return False
                finally:
                    os.chdir(original_cwd)
            else:
                print("✅ BERT and SSL models already loaded")
                return True
                
        except Exception as e:
            print(f"❌ Error in manual model loading: {e}")
            return False
    
    def ensure_models_loaded(self):
        """Ensure models are loaded before synthesis"""
        
        # First verify model files exist
        if not self._verify_model_files_exist():
            print("❌ Model files missing, cannot reload models")
            return
            
        try:
            import GPT_SoVITS.inference_webui as inference_webui
            
            # Check if main models are loaded and accessible
            models_loaded = True
            missing_models = []
            
            # Check critical models that must be loaded for synthesis
            critical_models = ['vq_model', 't2s_model', 'hps', 'ssl_model', 'bert_model']
            
            for model_name in critical_models:
                if not (hasattr(inference_webui, model_name) and 
                       getattr(inference_webui, model_name) is not None):
                    models_loaded = False
                    missing_models.append(model_name)
            
            if missing_models:
                print(f"🔄 Missing models: {', '.join(missing_models)}")
                print("🔄 Reloading all models...")
                
                # Debug: Check current working directory and available functions
                print(f"🔍 Current working directory: {os.getcwd()}")
                print(f"🔍 GPT-SoVITS path: {self.gpt_sovits_path}")
                print(f"🔍 Available functions: change_gpt_weights={self.change_gpt_weights}, change_sovits_weights={self.change_sovits_weights}")
                
                # Change to GPT-SoVITS directory for model loading
                original_cwd = os.getcwd()
                os.chdir(self.gpt_sovits_path)
                print(f"🔍 Changed to directory: {os.getcwd()}")
                
                try:
                    # Reload both GPT and SoVITS models (this loads all necessary models)
                    # Make sure we're in the right directory context
                    print(f"🔄 Loading GPT model from: {self.gpt_model_path}")
                    gpt_result = self.change_gpt_weights(gpt_path=self.gpt_model_path)
                    print(f"🔄 GPT model load result: {type(gpt_result)}")
                    
                    print(f"🔄 Loading SoVITS model from: {self.sovits_model_path}")
                    # The sovits function is a generator, so we need to consume it
                    # Also provide default language parameters to avoid the prompt_text_update error
                    sovits_generator = self.change_sovits_weights(
                        sovits_path=self.sovits_model_path,
                        prompt_language="中文",  # Provide default Chinese
                        text_language="中文"     # Provide default Chinese
                    )
                    sovits_results = []
                    try:
                        for result in sovits_generator:
                            sovits_results.append(result)
                            print(f"🔄 SoVITS generator yielded: {type(result)}")
                    except Exception as gen_e:
                        print(f"🔄 SoVITS generator completed or error: {gen_e}")
                    
                    print("✅ Model loading functions called successfully!")
                    
                    # Give models a moment to load
                    import time
                    time.sleep(3)  # Increased wait time
                    
                    # Manually load BERT and SSL models if needed
                    self._manually_load_bert_ssl_models()
                    
                    # Verify models are actually loaded after reload
                    print("🔍 Verifying model state after reload...")
                    for model_name in critical_models:
                        if hasattr(inference_webui, model_name):
                            model_value = getattr(inference_webui, model_name)
                            if model_value is None:
                                print(f"⚠️ Warning: {model_name} is None after reload")
                            else:
                                print(f"✅ {model_name} successfully loaded (type: {type(model_value)})")
                        else:
                            print(f"⚠️ Warning: {model_name} attribute missing after reload")
                    
                finally:
                    os.chdir(original_cwd)
            else:
                print("✅ All critical models already loaded")
            
        except Exception as e:
            print(f"Warning: Could not check model status: {e}")
            # Try to reload anyway
            try:
                original_cwd = os.getcwd()
                os.chdir(self.gpt_sovits_path)
                
                try:
                    print(f"🔄 Emergency reload - GPT model: {self.gpt_model_path}")
                    gpt_result = self.change_gpt_weights(gpt_path=self.gpt_model_path)
                    print(f"🔄 Emergency GPT result: {type(gpt_result)}")
                    
                    print(f"🔄 Emergency reload - SoVITS model: {self.sovits_model_path}")
                    # Handle generator properly with default language parameters
                    sovits_generator = self.change_sovits_weights(
                        sovits_path=self.sovits_model_path,
                        prompt_language="中文",
                        text_language="中文"
                    )
                    try:
                        for result in sovits_generator:
                            pass  # Just consume the generator
                    except Exception as gen_e:
                        print(f"🔄 Emergency SoVITS generator: {gen_e}")
                    
                    print("✅ Emergency models reloaded!")
                    
                    # Wait a bit and check again
                    import time
                    time.sleep(3)
                    
                    # Also try manual BERT/SSL loading
                    self._manually_load_bert_ssl_models()
                    
                    try:
                        import GPT_SoVITS.inference_webui as inference_webui
                        for model_name in ['vq_model', 't2s_model', 'hps']:
                            if hasattr(inference_webui, model_name):
                                model_value = getattr(inference_webui, model_name)
                                if model_value is None:
                                    print(f"⚠️ Emergency check: {model_name} is still None")
                                else:
                                    print(f"✅ Emergency check: {model_name} loaded")
                    except Exception as check_e:
                        print(f"Could not verify emergency reload: {check_e}")
                        
                finally:
                    os.chdir(original_cwd)
            except Exception as reload_error:
                print(f"Emergency reload failed: {reload_error}")
                # Last resort - try to reinitialize everything
                try:
                    print("🔄 Attempting full reinitialization...")
                    self._setup_gpt_sovits()
                except Exception as init_error:
                    print(f"Full reinitialization failed: {init_error}")
    
    def cleanup_models(self):
        """Unload GPT-SoVITS models and free GPU memory (conservative cleanup during processing)"""
        try:
            import gc
            import torch
            
            # Change to GPT-SoVITS directory to access global variables
            original_cwd = os.getcwd()
            os.chdir(self.gpt_sovits_path)
            
            # Import the inference module to access global variables
            try:
                import GPT_SoVITS.inference_webui as inference_webui
                
                # Only clear non-essential models during processing to avoid errors
                # Keep essential models loaded to prevent "NoneType has no attribute 'model'" errors
                
                models_to_clear = [
                    'hifigan_model', 'bigvgan_model'  # Only clear vocoder models, they reload quickly
                ]
                
                # DO NOT clear during processing (causes NoneType errors):
                # - vq_model, t2s_model (main synthesis models)
                # - bert_model, ssl_model (text processing models)  
                # - hps, config (essential configuration)
                
                for model_name in models_to_clear:
                    if hasattr(inference_webui, model_name):
                        try:
                            model = getattr(inference_webui, model_name)
                            if model is not None:
                                if hasattr(model, 'cpu'):
                                    model = model.cpu()
                                if hasattr(model, 'to'):
                                    model = model.to('cpu')
                                del model
                            setattr(inference_webui, model_name, None)
                            print(f"🧹 {model_name} cleared")
                        except Exception as e:
                            print(f"Warning clearing {model_name}: {e}")
                            try:
                                setattr(inference_webui, model_name, None)
                            except:
                                pass
                        
            except Exception as e:
                print(f"Warning: Could not access GPT-SoVITS global variables: {e}")
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            # Light memory cleanup
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            print("🧹 Conservative model cleanup completed (kept essential models loaded).")
            
        except Exception as e:
            print(f"Warning: Model cleanup failed: {e}")
            # Fallback cleanup
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
    
    def __del__(self):
        """Destructor to ensure models are cleaned up when object is destroyed"""
        try:
            if hasattr(self, 'change_gpt_weights') and self.change_gpt_weights is not None:
                self.cleanup_models()
        except:
            pass  # Ignore errors during destruction
    
    def _get_language_name(self, lang_code):
        """Convert simple language codes to GPT-SoVITS language names"""
        language_map = {
            'ja': '日文',
            'en': '英文', 
            'zh': '中文',
            'ko': '韩文',
            'es': '西班牙文',
            'fr': '法文',
            'de': '德文',
            'it': '意大利文',
            'pt': '葡萄牙文',
            'ru': '俄文'
        }
        return language_map.get(lang_code.lower(), '英文')  # Default to English if not found
        
    def synthesize_translations(self, transcribed_segments, translated_segments, voice_samples_dir, audio_segments_dir, top_k, top_p, temperature, speed, prompt_language="ja", target_language="en", read_from_cache=False, cache_path=None):
        """
        Synthesize translated audio for all speakers
        
        Args:
            transcribed_segments: Dict with speaker transcriptions
            translated_segments: Dict with speaker translations  
            voice_samples_dir: Directory containing voice samples
            audio_segments_dir: Directory containing audio segments
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            temperature: Temperature for synthesis
            speed: Speed of synthesis
            prompt_language: Language of the reference audio ('ja', 'en', 'zh', etc.)
            target_language: Target language for synthesis ('ja', 'en', 'zh', etc.)
            read_from_cache: Whether to try loading from cache first
            cache_path: Path to cache file
            
        Returns:
            Dict with synthesis results including timing information
        """
        # Ensure models are loaded before synthesis
        self.ensure_models_loaded()
        
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.speed = speed
        self.prompt_language = prompt_language
        self.target_language = target_language
        # Try loading from cache first
        synthesis_results = read_cache(read_from_cache, cache_path)
        if synthesis_results:
            print(f"Using cached synthesis results from: {cache_path}")
            # Even with cache, cleanup models that were loaded during init
            self.cleanup_models()
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
                prompt_language=self.prompt_language,
                target_language=self.target_language
            )
            
            synthesis_results[speaker_id] = speaker_results
            
            # Light memory cleanup between speakers (non-disruptive)
            self._intermediate_cleanup()
            
        # Save to cache
        if cache_path:
            save_cache(cache_path, synthesis_results)
            print(f"Synthesis results cached to: {cache_path}")
        
        # Only do light cleanup during processing, save aggressive cleanup for end
        # self.cleanup_models()  # Disabled - was too aggressive
        
        # Only do aggressive cleanup at the very end
        self._final_memory_cleanup()
            
        return synthesis_results
    
    def _final_memory_cleanup(self):
        """Aggressive memory cleanup after ALL synthesis is complete"""
        try:
            import gc
            import torch
            
            print("🧹 Performing aggressive final memory cleanup...")
            
            # Change to GPT-SoVITS directory to access global variables
            original_cwd = os.getcwd()
            os.chdir(self.gpt_sovits_path)
            
            try:
                import GPT_SoVITS.inference_webui as inference_webui
                
                # NOW we can aggressively clear ALL models since synthesis is done
                all_models_to_clear = [
                    'vq_model', 't2s_model', 'hifigan_model', 'bigvgan_model',
                    'bert_model', 'ssl_model'
                ]
                
                for model_name in all_models_to_clear:
                    if hasattr(inference_webui, model_name):
                        try:
                            model = getattr(inference_webui, model_name)
                            if model is not None:
                                if hasattr(model, 'cpu'):
                                    model = model.cpu()
                                if hasattr(model, 'to'):
                                    model = model.to('cpu')
                                del model
                            setattr(inference_webui, model_name, None)
                            print(f"  🧹 Final cleanup: {model_name} cleared")
                        except Exception as e:
                            print(f"  ⚠️ Warning clearing {model_name}: {e}")
                            try:
                                setattr(inference_webui, model_name, None)
                            except:
                                pass
                
                # Keep only essential lightweight configuration:
                # 'hps', 'config', 'dict_language', 'tokenizer' - for next video
                        
            except Exception as e:
                print(f"  ⚠️ Could not access GPT-SoVITS global variables: {e}")
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            # Clear any remaining references
            if hasattr(self, 'get_tts_wav'):
                # Don't delete the function reference - we need it for reloading
                pass
            
            # Aggressive garbage collection
            for i in range(3):
                collected = gc.collect()
                if collected > 0:
                    print(f"  🧹 GC round {i+1}: Collected {collected} objects")
            
            # Multiple GPU cache clears
            if torch.cuda.is_available():
                for _ in range(5):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Check memory usage
                try:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                    print(f"  📊 GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
                except:
                    pass
                    
        except Exception as e:
            print(f"Final cleanup warning: {e}")
    
    def _intermediate_cleanup(self):
        """Very lightweight cleanup between speakers - only cache clearing"""
        try:
            import gc
            import torch
            
            # Only do minimal cleanup that won't interfere with loaded models
            # Light garbage collection
            gc.collect()
            
            # Clear GPU cache once (safe)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Intermediate cleanup warning: {e}")
    
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

    def _synthesize_text_chunks(self, speaker_id, segment_num, text_chunks, reference_wav, reference_text, inp_refs, prompt_language, target_language, speaker_output_dir, start_time, end_time, original_text):
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
                # Ensure models are loaded before each chunk synthesis
                self.ensure_models_loaded()
                
                # Adjust parameters for continuation chunks
                synthesis_params = {
                    'ref_wav_path': reference_wav,
                    'prompt_text': reference_text,
                    'prompt_language': self.i18n(self._get_language_name(prompt_language)),
                    'text': chunk_text,
                    'text_language': self.i18n(self._get_language_name(target_language)),
                    'how_to_cut': self.i18n("不切"),
                    'top_k': self.top_k,
                    'top_p': self.top_p,
                    'temperature': self.temperature,
                    'ref_free': False,
                    'speed': self.speed,
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

    def _synthesize_speaker_segments(self, speaker_id, reference_wav, reference_text, other_references, translations, prompt_language, target_language):
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
                    # Ensure models are loaded right before synthesis
                    self.ensure_models_loaded()
                    
                    synthesis_result = self.get_tts_wav(
                        ref_wav_path=reference_wav,
                        prompt_text=reference_text,
                        prompt_language=self.i18n(self._get_language_name(prompt_language)),
                        text=translated_text,
                        text_language=self.i18n(self._get_language_name(target_language)),
                        how_to_cut=self.i18n("不切"),
                        top_k=self.top_k,  # 15,
                        top_p=self.top_p,  # 0.7,
                        temperature=self.temperature,  # 1,
                        ref_free=False,
                        speed=self.speed,  # 1.1,
                        if_freeze=False,
                        inp_refs=inp_refs,
                        sample_steps=8,
                        if_sr=False,
                        pause_second=0.3
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
                    inp_refs, prompt_language, target_language, speaker_output_dir, start_time, end_time, original_text
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