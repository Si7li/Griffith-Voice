# 🗣️ Real-time Voice Translation Web Interface

## Quick Start Guide

### 1. Launch the Web Interface
```bash
./run_web_interface.sh
```

*Or for testing:*
```bash
./test_web_interface.sh
```

*Alternative options:*
```bash
./demo.sh                    # Demo with instructions
streamlit run streamlit_clean.py  # Direct launch (clean, no warnings)
```

### 2. Open Your Browser
Go to: **http://localhost:8501**

### 3. Upload and Process
1. **Upload Video**: Click "Browse files" and select your video (MP4, AVI, MOV, MKV)
2. **Adjust Parameters**: Use the sidebar to customize:
   - **Languages**: Source and target languages
   - **Voice Synthesis**: Top K, Top P, Temperature, Speed
   - **Speaker Diarization**: Clustering method, thresholds, etc.
3. **Process**: Click "🎬 Process Video" and wait for completion
4. **Download**: Get your translated video and audio files

## 🎛️ Parameter Guide

### Voice Synthesis Parameters
- **Top K** (1-50): Controls voice diversity
  - Higher = more varied/creative voices
  - **Recommended**: 15
  
- **Top P** (0.1-1.0): Controls randomness
  - Higher = more unpredictable speech
  - **Recommended**: 0.7
  
- **Temperature** (0.1-2.0): Controls creativity
  - Higher = more experimental pronunciation
  - **Recommended**: 1.0
  
- **Speed** (0.5-2.0): Speech speed multiplier
  - 1.0 = original speed, 1.1 = 10% faster
  - **Recommended**: 1.1

### Speaker Diarization Parameters
- **Min Duration Off** (0.1-3.0s): Minimum silence between speakers
  - Lower = captures quick dialogue
  - **Recommended**: 0.75s
  
- **Clustering Method**: How to group speech segments
  - **average**: Best for short clips with emotional speech
  - **centroid**: Best for long clips with consistent voices
  - **pooling**: Best for highly varied voices
  
- **Min Cluster Size** (1-20): Minimum segments per speaker
  - Lower = keeps minor characters separate
  - **Recommended**: 8
  
- **Threshold** (0.1-1.0): Speaker similarity tolerance
  - Higher = more strict (avoids mixing speakers)
  - Lower = more lenient (avoids splitting same speaker)
  - **Recommended**: 0.78

## 🌍 Supported Languages

**Available**: Japanese (ja), English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Russian (ru), Korean (ko), Chinese (zh)

**Popular Combinations**:
- Japanese → English (anime/movies)
- English → Spanish (content localization)
- Korean → English (K-dramas)

## 📋 Processing Modes

### 🚀 Full Processing Mode
- All ML dependencies loaded
- Complete voice translation pipeline
- Real voice cloning with original characteristics
- High-quality output

### 🧪 Demo Mode
- Limited dependencies available
- Simulation of processing steps
- Parameter configuration works
- No actual translation (yet)

## 🔧 Troubleshooting

### Common Issues

**"Running in demo mode"**
- Install missing dependencies: `pip install -r requirements.txt`
- Set up GPT-SoVITS models (see main documentation)
- Restart the web interface

**"File too large"**
- Maximum file size is 500MB
- Use shorter clips for faster processing
- Compress video before uploading

**"Processing failed"**
- Check video has clear audio
- Ensure source language is correct
- Try adjusting clustering threshold

**Poor translation quality**
- Lower temperature/top_p for consistency
- Adjust clustering method for your content type
- Use higher threshold for speaker separation

### Performance Tips
- Use videos under 5 minutes for best performance
- Ensure good audio quality in source video
- Close other applications during processing
- GPU recommended for faster synthesis

## 📁 File Structure
After processing, you'll find:
```
outputs/
├── output_audio.wav          # Extracted audio
├── vocals.wav               # Separated vocals  
├── no_vocals.wav           # Background music
├── final_translated_audio.wav # Final translated audio
├── mixed.wav               # Audio + background
└── output.mp4              # Final translated video
```

## 🆘 Need Help?

1. **Check the tabs** in the web interface for detailed parameter explanations
2. **Read the main README** for complete setup instructions
3. **Check processing mode** - demo vs full processing
4. **Verify file format** - use supported video formats only

## 🎯 Best Practices

### For Best Results:
- **Clear audio**: Minimize background noise
- **Good speech**: Avoid mumbling or very fast speech  
- **Proper language**: Set correct source language
- **Reasonable length**: Under 10 minutes recommended

### Parameter Tuning:
- **Conservative first**: Use default parameters initially
- **One at a time**: Change one parameter and test
- **Content-specific**: Adjust clustering for your video type
- **Voice quality**: Lower temperature for more consistent voices

---

**Ready to translate? Run `./run_web_interface.sh` and start uploading! 🚀**
