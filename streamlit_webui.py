import warnings
import os
import sys

# Suppress common ML library warnings
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchaudio.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set environment variables to reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import streamlit as st
import tempfile
import yaml
import time
from pathlib import Path

# Set up page config first
st.set_page_config(
    page_title="VidDub",
    page_icon="💮",
    layout="wide"
)

# Flag to indicate if full processing is available
FULL_PROCESSING_AVAILABLE = False

# Try to import processing modules with better error handling
def load_processing_modules():
    global FULL_PROCESSING_AVAILABLE
    
    try:
        # Set torchaudio backend before importing other modules
        try:
            import torchaudio
            # Use the recommended backend dispatch method
            torchaudio.set_audio_backend("soundfile")
        except:
            pass
            
        # Import processing modules
        from extract_audio.extract_audio import ExtractAudio
        from separate_audio.separate_audio import SeparateAudio
        from diarize_audio.diarize_audio import AudioDiarization
        from extract_segments.extract_segments import SegmentExtractor
        from transcribe_audio_segments.transcribe_audio_segments import AudioTranscriber
        from translate_segments.translate_segments import SegmentsTranslator
        from sample_segments.sample_segments import SegmentsSampler
        from synthensize_translations.synthensize_translations import TranslationsSynthensizer, force_cleanup_gpt_sovits
        from assemble_translations.assemble_translations import AudioAssembler
        from apply_video_no_vocals.apply_video_no_vocals import VideoNoVocalsApplier
        
        FULL_PROCESSING_AVAILABLE = True
        return True, "✅ Full processing pipeline loaded successfully!"
        
    except ImportError as e:
        return False, f"⚠️ Running in demo mode - some dependencies not available"
    except Exception as e:
        return False, f"⚠️ Running in demo mode - {str(e)[:100]}..."

def load_config():
    """Load configuration from YAML file"""
    try:
        config_path = "configs/config.yaml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {
            "pipeline": {
                "params": {
                    "segmentation": {"min_duration_off": 0.75},
                    "clustering": {"method": "average", "min_cluster_size": 8, "threshold": 0.78}
                }
            }
        }

def save_config(config):
    """Save configuration to YAML file"""
    try:
        config_path = "configs/config.yaml"
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving config: {e}")
        return False

def create_temp_directories():
    """Create temporary directories for processing"""
    temp_dir = tempfile.mkdtemp(prefix="streamlit_translation_")
    
    dirs = {
        'inputs': os.path.join(temp_dir, 'inputs'),
        'outputs': os.path.join(temp_dir, 'outputs'),
        'audio_segments': os.path.join(temp_dir, 'outputs', 'audio_segments'),
        'voice_samples': os.path.join(temp_dir, 'outputs', 'voice_samples'),
        'translated_outputs': os.path.join(temp_dir, 'outputs', 'translated_outputs'),
        'caches': os.path.join(temp_dir, 'caches')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return temp_dir, dirs

def process_video_full(video_file, parameters, progress_callback=None):
    """Full video processing with actual ML pipeline"""
    try:
        # Clear output directories before starting new processing
        clear_output_directories()
        
        if progress_callback:
            progress_callback(0.05, "Output directories cleared, creating temp directories...")
        
        # Create temporary directories
        temp_dir, dirs = create_temp_directories()
        
        # Save uploaded video
        input_video_path = os.path.join(dirs['inputs'], 'input_video.mp4')
        with open(input_video_path, 'wb') as f:
            f.write(video_file.read())
        
        if progress_callback:
            progress_callback(0.1, "Video uploaded, extracting audio...")
        
        # Update config with new parameters
        config = load_config()
        config['pipeline']['params']['segmentation']['min_duration_off'] = parameters['min_duration_off']
        config['pipeline']['params']['clustering']['method'] = parameters['clustering_method']
        config['pipeline']['params']['clustering']['min_cluster_size'] = parameters['min_cluster_size']
        config['pipeline']['params']['clustering']['threshold'] = parameters['threshold']
        save_config(config)
        
        # Import modules locally to avoid global import issues
        from extract_audio.extract_audio import ExtractAudio
        from separate_audio.separate_audio import SeparateAudio
        from diarize_audio.diarize_audio import AudioDiarization
        from extract_segments.extract_segments import SegmentExtractor
        from transcribe_audio_segments.transcribe_audio_segments import AudioTranscriber
        from translate_segments.translate_segments import SegmentsTranslator
        from sample_segments.sample_segments import SegmentsSampler
        from synthensize_translations.synthensize_translations import TranslationsSynthensizer, force_cleanup_gpt_sovits
        from assemble_translations.assemble_translations import AudioAssembler
        from apply_video_no_vocals.apply_video_no_vocals import VideoNoVocalsApplier
        from utils import comprehensive_final_cleanup
        
        # Step 1: Extract audio
        audio_extractor = ExtractAudio(input_video_path)
        output_audio_path = os.path.join(dirs['outputs'], 'output_audio.wav')
        result_audio = audio_extractor.extract_audio(output_audio_path, read_from_cache=False)
        
        if progress_callback:
            progress_callback(0.2, "Audio extracted, separating vocals...")
        
        # Step 2: Separate audio
        audio_separator = SeparateAudio(result_audio)
        result_audio_separated = audio_separator.separate_audio(read_from_cache=False)
        vocals = result_audio_separated['vocals']
        no_vocals = result_audio_separated['music']
        
        if progress_callback:
            progress_callback(0.3, "Audio separated, performing speaker diarization...")
        
        # Step 3: Diarize audio
        audio_diarizer = AudioDiarization(vocals)
        diarization = audio_diarizer.diarize_audio(read_from_cache=False)
        
        if progress_callback:
            progress_callback(0.4, "Diarization complete, extracting segments...")
        
        # Step 4: Extract segments
        segments_extractor = SegmentExtractor(vocals, diarization)
        extracted = segments_extractor.extract_segments(dirs['audio_segments'])
        
        if progress_callback:
            progress_callback(0.5, "Segments extracted, transcribing audio...")
        
        # Step 5: Transcribe audio segments
        audio_transcriber = AudioTranscriber("small")
        transcribed_segments = audio_transcriber.transcribe_folder(
            segments_folder=dirs['audio_segments'],
            diarization_data=diarization,
            language=parameters['source_language'],
            read_from_cache=False
        )
        
        if progress_callback:
            progress_callback(0.6, "Transcription complete, translating segments...")
        
        # Step 6: Translate segments
        segments_translator = SegmentsTranslator()
        translated_segments = segments_translator.translate_segments(
            transcribed_segments=transcribed_segments,
            diarization_essensials=diarization,
            source_lang=parameters['source_language'],
            target_lang=parameters['target_language'],
            read_from_cache=False
        )
        
        if progress_callback:
            progress_callback(0.7, "Translation complete, sampling voice segments...")
        
        # Step 7: Sample segments
        segments_sampler = SegmentsSampler(dirs['audio_segments'], dirs['voice_samples'])
        audio_samples = segments_sampler.merge(transcribed_data=translated_segments, read_from_cache=False)
        
        if progress_callback:
            progress_callback(0.8, "Voice sampling complete, synthesizing translations...")
        
        # Step 8: Synthesize translations
        translations_synthesizer = TranslationsSynthensizer()
        synthesis_results = translations_synthesizer.synthesize_translations(
            transcribed_segments=transcribed_segments,
            translated_segments=translated_segments,
            voice_samples_dir=dirs['voice_samples'],
            audio_segments_dir=dirs['audio_segments'],
            top_k=parameters['top_k'],
            top_p=parameters['top_p'],
            temperature=parameters['temperature'],
            speed=parameters['speed'],
            prompt_language=parameters['source_language'],
            target_language=parameters['target_language'],
            read_from_cache=False
        )
        
        # Explicitly delete synthesizer to ensure GPU cleanup
        del translations_synthesizer
        
        # Force cleanup of GPT-SoVITS models
        force_cleanup_gpt_sovits()
        
        if progress_callback:
            progress_callback(0.9, "Synthesis complete, assembling final video...")
        
        # Step 9: Assemble audio
        audio_assembler = AudioAssembler(input_video_path)
        final_audio_path = os.path.join(dirs['outputs'], 'final_translated_audio.wav')
        final_audio = audio_assembler.assemble_audio(
            synthesis_results=synthesis_results,
            output_path=final_audio_path,
            read_from_cache=False
        )
        
        # Step 10: Apply video with no vocals
        final_video_path = os.path.join(dirs['outputs'], 'output.mp4')
        mixed_audio_path = os.path.join(dirs['outputs'], 'mixed.wav')
        
        video_no_vocals_applier = VideoNoVocalsApplier(
            final_translated_audio=final_audio,
            no_vocals_path=no_vocals,
            input_video=input_video_path
        )
        video_no_vocals_applier.process(
            mixed_audio_out=mixed_audio_path,
            final_video_out=final_video_path,
            voice_volume=parameters['voice_volume'],
            background_volume=parameters['background_volume'],
            master_volume=parameters['master_volume']
        )
        
        if progress_callback:
            progress_callback(1.0, "Processing complete!")
        
        # Final comprehensive cleanup to ensure all models are unloaded
        comprehensive_final_cleanup()
        
        return {
            'success': True,
            'final_video': final_video_path,
            'final_audio': final_audio_path,
            'mixed_audio': mixed_audio_path,
            'temp_dir': temp_dir,
            'transcribed_segments': transcribed_segments,
            'translated_segments': translated_segments
        }
        
    except Exception as e:
        # Cleanup on error too
        try:
            comprehensive_final_cleanup()
        except:
            pass
        return {
            'success': False,
            'error': str(e),
            'temp_dir': temp_dir if 'temp_dir' in locals() else None
        }

def process_video_demo(video_file, parameters, progress_callback=None):
    """Demo processing simulation"""
    temp_dir = tempfile.mkdtemp(prefix="demo_translation_")
    
    try:
        # Save uploaded video
        input_video_path = os.path.join(temp_dir, 'input_video.mp4')
        with open(input_video_path, 'wb') as f:
            f.write(video_file.read())
        
        # Simulate processing steps
        steps = [
            "🎬 Video uploaded successfully",
            "🎵 Extracting audio track...",
            "🎤 Separating vocals from music...",
            "👥 Identifying different speakers...",
            "✂️ Extracting speech segments...",
            "📝 Transcribing speech to text...",
            "🌍 Translating to target language...",
            "🎭 Sampling voice characteristics...",
            "🗣️ Synthesizing translated speech...",
            "🎧 Assembling final audio...",
            "🎬 Creating final video..."
        ]
        
        for i, step in enumerate(steps):
            if progress_callback:
                progress = (i + 1) / len(steps)
                progress_callback(progress, step)
            time.sleep(0.3)
        
        return {
            'success': True,
            'message': 'Demo processing completed!',
            'temp_dir': temp_dir,
            'input_path': input_video_path
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'temp_dir': temp_dir
        }

def cleanup_temp_files():
    """Clean up temporary files and directories."""
    import glob
    import shutil
    
    patterns = ['temp_separation*', 'streamlit_translation_*', 'demo_translation_*']
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                except OSError:
                    pass  # Directory might be in use

def clear_output_directories():
    """Clear all output directories before processing a new video"""
    import shutil
    
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
    cleanup_temp_files()
    # Load processing modules
    success, message = load_processing_modules()
    
    # Show loading status in sidebar initially
    with st.sidebar:
        if success:
            st.success("🚀 Full Processing Available")
        else:
            st.info("🧪 Demo Mode Active")
    
    st.title("💮 VidDub")
    st.markdown("Upload a video, adjust parameters, and get a translated version with voice cloning!")
    
    # Custom CSS for Professional Dark Mode
    st.markdown("""
    <style>
    /* Dark Mode Base */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main .block-container {
        background-color: #0e1117;
        color: #fafafa;
        padding-top: 2rem;
    }
    
    /* Text Colors - Light text on dark background */
    .stMarkdown {
        color: #fafafa !important;
    }
    .stMarkdown p {
        color: #fafafa !important;
    }
    .stMarkdown li {
        color: #fafafa !important;
    }
    .stMarkdown strong {
        color: #ff99b7 !important;
        font-weight: 600;
    }
    
    /* Headers - Bright colors for hierarchy */
    h1 {
        color: #ff99b7 !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    h2, h3, h4, h5, h6 {
        color: #81c784 !important;
    }
    
    /* Sidebar Dark Mode */
    .css-1d391kg {
        background-color: #1e1e2e;
    }
    .css-1d391kg .stMarkdown {
        color: #fafafa !important;
    }
    
    /* Upload Section - Blue theme */
    .upload-box {
        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
        color: white !important;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(33, 150, 243, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .upload-box h3 {
        color: white !important;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .upload-box p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem;
        margin-bottom: 0;
    }
    
    /* Results Section - Green theme */
    .results-box {
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        color: white !important;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .results-box h3 {
        color: white !important;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .results-box p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1rem;
        margin-bottom: 0;
    }
    
    /* Info Boxes - Dark blue theme */
    .stAlert > div {
        background-color: #1a237e !important;
        color: #e3f2fd !important;
        border: 1px solid #3f51b5 !important;
        border-radius: 8px;
    }
    .stAlert > div > div {
        color: #e3f2fd !important;
    }
    .stAlert p {
        color: #e3f2fd !important;
    }
    .stAlert code {
        background-color: #0d47a1 !important;
        color: #e3f2fd !important;
        border: 1px solid #1976d2 !important;
        border-radius: 4px;
        padding: 2px 4px;
    }
    
    /* Progress Bar */
    .stProgress .st-bp {
        background-color: #ff99b7;
    }
    
    /* Buttons - Red gradient for action */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #ee5a24, #ff6b6b);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
    }
    
    /* Column content - ensure all text is visible */
    .element-container .stMarkdown {
        color: #fafafa !important;
    }
    .element-container .stMarkdown p {
        color: #fafafa !important;
    }
    .element-container .stMarkdown strong {
        color: #ff99b7 !important;
    }
    
    /* Hide warnings */
    .stAlert[data-baseweb="notification"] {
        display: none;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #1b5e20 !important;
        color: #c8e6c9 !important;
    }
    .stError {
        background-color: #b71c1c !important;
        color: #ffcdd2 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for parameters
    st.sidebar.header("⚙️ Processing Parameters")
    
    # Load current config
    config = load_config()
    
    # Language selection
    st.sidebar.subheader("🌍 Language Settings")
    languages = {
        "ja": "Japanese (日本語)",
        "en": "English",
#        "es": "Spanish (Español)",
#        "fr": "French (Français)",
#        "de": "German (Deutsch)",
#        "it": "Italian (Italiano)",
#        "pt": "Portuguese (Português)",
#        "ru": "Russian (Русский)",
        "ko": "Korean (한국어)",
#        "zh": "Chinese (中文)"
    }
    
    source_language = st.sidebar.selectbox(
        "Source Language",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=0,
        help="Language of the input video"
    )
    
    target_language = st.sidebar.selectbox(
        "Target Language",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=1,
        help="Language to translate to"
    )
    
    # Voice synthesis parameters
    st.sidebar.subheader("🎤 Voice Synthesis Parameters")
    top_k = st.sidebar.slider("Top K", 1, 50, 15, help="Controls diversity of voice generation")
    top_p = st.sidebar.slider("Top P", 0.1, 1.0, 0.7, 0.1, help="Controls randomness")
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1, help="Controls creativity")
    speed = st.sidebar.slider("Speed", 0.5, 2.0, 1.1, 0.1, help="Speech speed multiplier")
    
    # Audio mixing parameters
    st.sidebar.subheader("🎵 Audio Mixing Parameters")
    voice_volume = st.sidebar.slider("Voice Volume", 0.0, 2.0, 1.0, 0.1, help="Volume level for translated voice (1.0 = normal)")
    background_volume = st.sidebar.slider("Background Volume", 0.0, 2.0, 0.3, 0.1, help="Volume level for background music/sounds (0.3 = 30% - recommended)")
    master_volume = st.sidebar.slider("Master Volume", 0.5, 3.0, 1.2, 0.1, help="Overall output amplification (1.2 = 20% boost - recommended for better audibility)")
    
    # Diarization parameters
    st.sidebar.subheader("👥 Speaker Diarization Parameters")
    min_duration_off = st.sidebar.slider(
        "Min Duration Off",
        0.1, 3.0,
        config['pipeline']['params']['segmentation']['min_duration_off'],
        0.05,
        help="Minimum silence between speech segments (seconds)"
    )
    
    clustering_method = st.sidebar.selectbox(
        "Clustering Method",
        ["average", "centroid", "complete", "median", "single", "ward"],
        index=["average", "centroid", "complete", "median", "single", "ward"].index(
            config['pipeline']['params']['clustering']['method'] if config['pipeline']['params']['clustering']['method'] in ["average", "centroid", "complete", "median", "single", "ward"] else "ward"
        ),
        help="Linkage method for speaker clustering:\n• ward: Best overall (minimizes variance)\n• average: Good default (balanced)\n• complete: For distinct speakers\n• centroid: Fast, geometric approach\n• median: Robust to outliers\n• single: Chain-like clusters"
    )
    
    min_cluster_size = st.sidebar.slider(
        "Min Cluster Size",
        1, 20,
        config['pipeline']['params']['clustering']['min_cluster_size'],
        help="Minimum segments required to form a speaker cluster"
    )
    
    threshold = st.sidebar.slider(
        "Threshold",
        0.1, 1.0,
        config['pipeline']['params']['clustering']['threshold'],
        0.01,
        help="Speaker similarity threshold (higher = more strict)"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-box">
        <h3>📁 Upload Video</h3>
        <p>Choose a video file to translate</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload the video you want to translate"
        )
        
        if uploaded_file is not None:
            # Validate file
            file_size_mb = uploaded_file.size / 1024 / 1024
            if file_size_mb > 500:
                st.error("❌ File too large (max 500MB)")
            else:
                st.success(f"✅ Uploaded: {uploaded_file.name} ({file_size_mb:.1f} MB)")
                st.video(uploaded_file)
    
    with col2:
        st.markdown("""
        <div class="results-box">
        <h3>🎬 Results</h3>
        <p>Processing results will appear here</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'processing_result' not in st.session_state:
            st.info("Upload a video and click 'Process Video' to see results here")
        else:
            result = st.session_state.processing_result
            if result['success']:
                st.success("✅ Processing completed successfully!")
                
                if FULL_PROCESSING_AVAILABLE and 'final_video' in result:
                    # Show actual results
                    if os.path.exists(result['final_video']):
                        st.video(result['final_video'])
                        
                        with open(result['final_video'], 'rb') as f:
                            st.download_button(
                                label="📥 Download Translated Video",
                                data=f.read(),
                                file_name=f"translated_{uploaded_file.name if uploaded_file else 'video.mp4'}",
                                mime="video/mp4"
                            )
                        
                        if 'final_audio' in result and os.path.exists(result['final_audio']):
                            with open(result['final_audio'], 'rb') as f:
                                st.download_button(
                                    label="🎵 Download Audio Only",
                                    data=f.read(),
                                    file_name="translated_audio.wav",
                                    mime="audio/wav"
                                )
                else:
                    # Demo mode results
                    st.info("🧪 Demo completed! In full mode, the translated video would appear here.")
                    st.download_button(
                        label="📥 Download Demo File",
                        data=b"Demo translation result",
                        file_name="demo_result.txt",
                        mime="text/plain"
                    )
            else:
                st.error(f"❌ Processing failed: {result['error']}")
    
    # Process button
    st.markdown("---")
    
    if st.button("🎬 Process Video", disabled=(uploaded_file is None), type="primary"):
        cleanup_temp_files()
        clear_output_directories()  # Clear outputs before new processing
        if uploaded_file is not None:
            # Update config
            config['pipeline']['params']['segmentation']['min_duration_off'] = min_duration_off
            config['pipeline']['params']['clustering']['method'] = clustering_method
            config['pipeline']['params']['clustering']['min_cluster_size'] = min_cluster_size
            config['pipeline']['params']['clustering']['threshold'] = threshold
            
            if save_config(config):
                st.success("✅ Configuration updated")
            
            parameters = {
                'source_language': source_language,
                'target_language': target_language,
                'top_k': top_k,
                'top_p': top_p,
                'temperature': temperature,
                'speed': speed,
                'voice_volume': voice_volume,
                'background_volume': background_volume,
                'master_volume': master_volume,
                'min_duration_off': min_duration_off,
                'clustering_method': clustering_method,
                'min_cluster_size': min_cluster_size,
                'threshold': threshold
            }
            
            # Show parameters being used
            with st.expander("📋 View Processing Parameters"):
                st.json(parameters)
            
            # Choose processing method
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress, message):
                progress_bar.progress(progress)
                status_text.text(message)
            
            # Process the video
            with st.spinner(f"Processing video in {'Full' if FULL_PROCESSING_AVAILABLE else 'Demo'} mode..."):
                if FULL_PROCESSING_AVAILABLE:
                    result = process_video_full(uploaded_file, parameters, update_progress)
                else:
                    result = process_video_demo(uploaded_file, parameters, update_progress)
                
                st.session_state.processing_result = result
                
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show completion message
                if result['success']:
                    st.balloons()
                    st.success("🎉 Processing complete!")
                else:
                    st.error("❌ Processing failed!")
                
                # Refresh to show results
                st.rerun()
    
    # Quick info section
    st.markdown("---")
    st.markdown("### 📋 Quick Info")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        **🌍 Translation**
        - From: {languages[source_language]}
        - To: {languages[target_language]}
        """)
    
    with col2:
        st.markdown(f"""
        **🎤 Voice Settings**
        - Top K: {top_k}
        - Temperature: {temperature}
        - Speed: {speed}x
        - Voice Vol: {voice_volume}x
        - BG Vol: {background_volume}x
        - Master Vol: {master_volume}x
        """)
    
    with col3:
        st.markdown(f"""
        **👥 Speaker Detection**
        - Method: {clustering_method}
        - Threshold: {threshold}
        - Min Size: {min_cluster_size}
        """)
    
    # Status and help
    if not FULL_PROCESSING_AVAILABLE:
        st.info("""
        🚧 **Demo Mode Active**: To enable full processing, install all dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        Then restart the web interface.
        """)
    
    # Footer
#    st.markdown("---")
#    st.markdown("Built with ❤️ using Streamlit • Powered by Whisper, GPT-SoVITS, and pyannote.audio")

if __name__ == "__main__":
    main()
