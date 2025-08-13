# Real-time Voice Translation Web Interface

A user-friendly web interface for translating video content with voice cloning capabilities.

## Features

- 🎬 **Video Upload**: Support for MP4, AVI, MOV, and MKV formats
- 🗣️ **Voice Cloning**: Maintains original speaker characteristics in translations
- 🌍 **Multi-language**: Supports Japanese, English, Spanish, French, German, Italian, Portuguese, Russian, Korean, and Chinese
- ⚙️ **Customizable Parameters**: Fine-tune voice synthesis and speaker diarization
- 📊 **Real-time Progress**: Track processing status with detailed progress updates
- 📝 **Transcription View**: See original and translated text side by side
- 📱 **Browser Playback**: Preview results directly in your browser
- 📥 **Easy Download**: Get translated video and audio files

## Quick Start

1. **Launch the Web Interface**:
   ```bash
   ./run_web_interface.sh
   ```
   
2. **Open Your Browser**: Go to `http://localhost:8501`

3. **Upload & Process**: 
   - Upload your video file
   - Adjust parameters as needed
   - Click "Process Video"
   - Wait for processing to complete
   - View and download results

## Parameters Guide

### Voice Synthesis Parameters

- **Top K** (1-50): Controls diversity of voice generation
  - Higher values = more diverse/creative output
  - Recommended: 15

- **Top P** (0.1-1.0): Controls randomness in voice generation
  - Higher values = more random/varied output
  - Recommended: 0.7

- **Temperature** (0.1-2.0): Controls creativity in voice generation
  - Higher values = more creative/unpredictable output
  - Recommended: 1.0

- **Speed** (0.5-2.0): Speech speed multiplier
  - 1.0 = normal speed, 1.1 = slightly faster
  - Recommended: 1.1

### Speaker Diarization Parameters

- **Min Duration Off** (0.1-3.0 seconds): Minimum silence between speech segments
  - Lower values = captures quick back-and-forth dialogue
  - Higher values = more conservative segment splitting
  - Recommended: 0.75

- **Clustering Method**: Method for grouping segments into speakers
  - **average**: Best for short clips with tone changes
  - **centroid**: Best for long clips with consistent voices  
  - **pooling**: Best for varied voices or extreme tone shifts

- **Min Cluster Size** (1-20): Minimum segments required to form a speaker cluster
  - Lower values = keeps minor characters as separate speakers
  - Higher values = more conservative speaker identification
  - Recommended: 8

- **Threshold** (0.1-1.0): Speaker similarity threshold
  - Higher values = more strict (risk of splitting same speaker)
  - Lower values = more merging (risk of mixing speakers)
  - Recommended: 0.78

## Supported Languages

**Source Languages**: Japanese (ja), English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Russian (ru), Korean (ko), Chinese (zh)

**Target Languages**: Same as source languages

## Processing Pipeline

1. **Audio Extraction**: Extract audio track from video
2. **Audio Separation**: Separate vocals from background music
3. **Speaker Diarization**: Identify different speakers
4. **Segment Extraction**: Split audio into speaker segments
5. **Transcription**: Convert speech to text using Whisper
6. **Translation**: Translate text to target language
7. **Voice Sampling**: Extract voice characteristics per speaker
8. **Synthesis**: Generate translated speech with original voices
9. **Assembly**: Combine translated segments
10. **Final Video**: Mix with background audio and apply to video

## Troubleshooting

### Common Issues

1. **"Module not found" errors**: Make sure you've activated the virtual environment and installed all dependencies
2. **Processing fails**: Check that input video has clear audio and speech
3. **Poor voice quality**: Try adjusting synthesis parameters (lower temperature/top_p for more consistent output)
4. **Speaker confusion**: Adjust clustering threshold and method for your content type

### Performance Tips

- Use shorter videos (< 5 minutes) for faster processing
- Ensure good audio quality in source video
- Close other applications to free up memory during processing
- For long videos, consider processing in segments

## File Structure

```
outputs/
├── output_audio.wav          # Extracted audio
├── vocals.wav               # Separated vocals
├── no_vocals.wav           # Background music
├── audio_segments/         # Individual speaker segments
├── voice_samples/          # Voice characteristic samples
├── translated_outputs/     # Synthesized translations
├── final_translated_audio.wav  # Final assembled audio
├── mixed.wav              # Mixed audio (translation + background)
└── output.mp4             # Final translated video
```

## Advanced Usage

### Custom Configuration

Edit `configs/config.yaml` to modify default diarization settings or `configs/web_config.json` for web interface settings.

### Command Line Processing

For batch processing or automation, use the original `main.py` script:

```bash
python main.py
```

## Requirements

- Python 3.8+
- FFmpeg
- GPU recommended for faster processing
- 4GB+ RAM
- Internet connection (for some models)

## Support

For issues or questions, please check the main project documentation or create an issue in the repository.
