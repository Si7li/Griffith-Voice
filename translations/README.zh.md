<div align="center">

<img src="/docs/logo.png" alt="Griffith Voice Logo" height="500" width="500">

# Griffith Voice — AI 语音克隆和配音工具

[**English**](/README.md)｜[**简体中文**](/translations/README.zh.md)｜[**日本語**](/translations/README.ja.md)

</div>

## 🌟 概述

Griffith Voice 是一个尖端工具，可以实时翻译和合成语音。它通过结合先进的 AI 模型进行转录、翻译和语音合成，实现跨语言的无缝通信。

主要功能：
- 🎙️ 使用 Whisper 进行实时转录
- 🌍 由 GPT 模型驱动的多语言翻译
- 🗣️ 使用 GPT-SoVITS 进行高质量语音合成
- 🚀 基于 Streamlit 的 Web 界面，便于交互
- 📊 详细的日志记录和进度跟踪

与类似项目的不同之处：**专注于实时处理和高质量语音合成。**

## 🎥 演示

### 实时翻译和配音

探索 Griffith Voice 在多语言实时翻译和配音中的能力。每个演示展示：
- 准确的语音转录。
- 无缝翻译成多种语言。
- 使用自然声音的高质量语音合成。

<table>
<tr>
<td width="25%">

#### 英语
https://github.com/user-attachments/assets/68ccfa11-ed76-4f0e-8a33-bb8dcea31007

</td>
<td width="25%">

#### 日语
https://github.com/user-attachments/assets/46e88ebb-a52f-4922-8391-cf1d2b2dc751

</td>
<td width="25%">

#### 韩语
https://github.com/user-attachments/assets/60194f04-7296-40d0-8501-eaa291e94793

</td>
<td width="25%">

#### 中文
https://github.com/user-attachments/assets/9d65ace4-115d-4e9b-b6f4-b96ae6ee6e0b

</td>
</tr>
</table>

### 语言支持

**支持的输入和输出语言：**

🇺🇸 英语 | 🇯🇵 日语 | 🇰🇷 韩语 | 🇨🇳 中文

**即将支持的输入和输出语言：**

🇪🇭 阿拉伯语 | 🇫🇷 法语 | 🇷🇺 俄语 | 🇩🇪 德语

**翻译支持所有语言，语音合成取决于所选的 TTS 模型。**

## 安装

> **注意：** 确保您已安装 Python 3.10+ 和 FFmpeg。

1. 克隆仓库

```bash
git clone https://github.com/Si7li/Griffith-Voice.git
cd Griffith-Voice
```

2. 创建并激活虚拟环境

```bash
python3 -m venv env
source env/bin/activate
```

3. 安装依赖项

```bash
pip install -r requirements.txt
```

4. 启动应用程序

```bash
python streamlit_webui.py
```

## APIs
Griffith Voice 支持：
- 转录：Whisper
- 翻译：GPT 模型
- 语音合成：GPT-SoVITS

## 为什么选择 Griffith Voice？

- **针对低 VRAM GPU 进行了优化**：即使在仅有 4GB VRAM 的 GPU 上也能流畅运行，确保广泛用户的可访问性。
- **高效的内存使用**：设计旨在最大限度地减少 GPU 内存消耗，同时不影响性能。
- **实时处理**：提供快速的转录、翻译和合成，非常适合实时应用。
- **可扩展性**：无论您使用的是高端工作站还是普通设备，该工具都能适应您的硬件能力。

这些功能使 Griffith Voice 成为实时通信和内容创作的多功能高效解决方案。

## 📄 许可证

本项目已获得 Apache 2.0 许可证。

## 📬 联系我

- 在 GitHub 上提交 [问题](https://github.com/Si7li/Griffith-Voice/issues) 或 [拉取请求](https://github.com/Si7li/Griffith-Voice/pulls)
- 电子邮件：mohamedkhalil.sahli@enicar.ucar.tn

---

<p align="center">如果您觉得这个项目有帮助，请给它一个 ⭐️！</p>
