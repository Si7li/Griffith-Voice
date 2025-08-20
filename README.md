<div align="center">

<img src="/docs/logo.png" alt="Griffith Voice Logo" height="140">

# Griffith Voice — AI Voice Cloner & Dubber


[**English**](/README.md)｜[**简体中文**](/translations/README.zh.md)｜[**日本語**](/translations/README.ja.md)

</div>

## 🌟 Overview

Griffith Voice is a cutting-edge tool for translating and synthesizing voice in real-time. It enables seamless communication across language barriers by combining advanced AI models for transcription, translation, and voice synthesis.

Key features:
- 🎙️ Real-time transcription with Whisper
- 🌍 Multi-language translation powered by GPT models
- 🗣️ High-quality voice synthesis using GPT-SoVITS
- 🚀 Streamlit-based web interface for easy interaction
- 📊 Detailed logging and progress tracking

Difference from similar projects: **Focus on real-time processing and high-quality voice synthesis.**

## 🎥 Demo

### Real-time Translation and Dubbing

Explore Griffith Voice's capabilities in real-time translation and dubbing for multiple languages. Each demo showcases:
- Accurate transcription of speech.
- Seamless translation into multiple languages.
- High-quality voice synthesis for natural-sounding output.

<table>
<tr>
<td width="25%">

#### English
<video controls width="300" height="200">
  <source src="inputs/videos/english_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="25%">

#### Japanese
<video controls width="300" height="200">
  <source src="inputs/videos/japanese_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="25%">

#### Korean
<video controls width="300" height="200">
  <source src="inputs/videos/korean_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="25%">

#### Chinese
<video controls width="300" height="200">
  <source src="inputs/videos/chinese_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
</tr>
</table>

<table>
<tr>
<td width="25%">

<video controls width="300" height="200">
  <source src="inputs/videos/english_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="25%">

<video controls width="300" height="200">
  <source src="inputs/videos/japanese_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="25%">

<video controls width="300" height="200">
  <source src="inputs/videos/korean_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="25%">

<video controls width="300" height="200">
  <source src="inputs/videos/chinese_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
</tr>
</table>

<table>
<tr>
<td width="25%">

<video controls width="300" height="200">
  <source src="inputs/videos/english_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="25%">

<video controls width="300" height="200">
  <source src="inputs/videos/japanese_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="25%">

<video controls width="300" height="200">
  <source src="inputs/videos/korean_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
<td width="25%">

<video controls width="300" height="200">
  <source src="inputs/videos/chinese_demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

</td>
</tr>
</table>

> **Note:** If any video does not play, ensure the corresponding file exists in the `inputs/videos/` directory or replace it with a valid YouTube link.


### Language Support

**Input Language Support:**

🇺🇸 English | 🇯🇵 Japanese | 🇰🇷 Korean | 🇨🇳 Chinese

**Translation supports all languages, while synthesis depends on the chosen TTS model.**

## Installation

> **Note:** Ensure you have Python 3.10+ and FFmpeg installed.

1. Clone the repository

```bash
git clone https://github.com/Si7li/Griffith-Voice.git
cd Griffith-Voice
```

2. Create and activate a virtual environment

```bash
python3 -m venv env
source env/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Start the application

```bash
python streamlit_webui.py
```

## APIs
Griffith Voice supports:
- Transcription: Whisper
- Translation: GPT models
- Voice Synthesis: GPT-SoVITS

## Why Choose Griffith Voice?

- **Optimized for Low VRAM GPUs**: Works seamlessly on GPUs with as little as 4GB VRAM, ensuring accessibility for a wide range of users.
- **Efficient Memory Usage**: Designed to minimize GPU memory consumption without compromising performance.
- **Real-time Processing**: Delivers fast transcription, translation, and synthesis, making it ideal for live applications.
- **Scalable Across Devices**: Whether you're using a high-end workstation or a modest setup, the tool adapts to your hardware capabilities.

These features make Griffith Voice a versatile and efficient solution for real-time communication and content creation.

## 📄 License

This project is licensed under the Apache 2.0 License.

## 📬 Contact Me

- Submit [Issues](https://github.com/Si7li/Griffith-Voice/issues) or [Pull Requests](https://github.com/Si7li/Griffith-Voice/pulls) on GitHub
- Email: team@griffith.io

---

<p align="center">If you find this project helpful, please give it a ⭐️!</p>