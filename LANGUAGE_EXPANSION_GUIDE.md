# Language Expansion Guide

## Overview
This guide explains how to restore additional language support to the voice translation interface when they are fully supported by your ML pipeline.

## Current Status
- **Supported**: Japanese, English
- **Removed**: Spanish, French, German, Italian, Portuguese, Russian, Korean, Chinese

## How to Add Languages Back

### Step 1: Update Language Dictionaries

In each Streamlit file, find the `languages` dictionary and add the desired languages back:

**Files to Update:**
- `streamlit_clean.py` (line ~447)
- `streamlit_final.py` (line ~283)
- `streamlit_simple.py` (line ~22 in `get_language_display_name` function)

**Original Full Language Dictionary:**
```python
languages = {
    "ja": "Japanese (日本語)",
    "en": "English",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "it": "Italian (Italiano)",
    "pt": "Portuguese (Português)",
    "ru": "Russian (Русский)",
    "ko": "Korean (한국어)",
    "zh": "Chinese (中文)"
}
```

### Step 2: Update Help Text

**In `streamlit_final.py`** (around line ~507):
```python
**🌍 Supported Languages:**
Japanese, English, Spanish, French, German, Italian, Portuguese, Russian, Korean, Chinese

**🚧 Coming Soon:**
# Remove this section when all languages are supported
```

**In `streamlit_simple.py`** (around line ~397):
```python
- **Languages:** 10+ languages including Japanese, English, Spanish, French, German, Italian, Portuguese, Russian, Korean, Chinese
```

### Step 3: Verify Backend Support

Before adding languages back, ensure your ML pipeline components support them:

1. **Whisper** - Check transcription quality for the target language
2. **Translation Model** - Verify translation quality between language pairs  
3. **GPT-SoVITS** - Ensure voice synthesis works for the target language
4. **Speaker Diarization** - Test that pyannote.audio works well with the language

### Step 4: Testing

After adding languages back:

1. **Test each language pair** individually
2. **Verify voice quality** in synthesis
3. **Check translation accuracy** 
4. **Test edge cases** (multiple speakers, background noise, etc.)

### Step 5: Update Documentation

When languages are fully supported:

1. Update `README.md` with supported languages
2. Update any help text or tooltips
3. Update configuration examples

## Language Codes Reference

| Code | Language | Native Name |
|------|----------|-------------|
| ja   | Japanese | 日本語 |
| en   | English  | English |
| es   | Spanish  | Español |
| fr   | French   | Français |
| de   | German   | Deutsch |
| it   | Italian  | Italiano |
| pt   | Portuguese | Português |
| ru   | Russian  | Русский |
| ko   | Korean   | 한국어 |
| zh   | Chinese  | 中文 |

## Notes

- Always test thoroughly before enabling a language in production
- Consider adding language-specific optimizations (voice model selection, etc.)
- Update error messages to be language-appropriate
- Consider adding more languages beyond the original 10 as your pipeline improves

## Quick Restore Command

To quickly restore all languages, you can search and replace in your IDE:

**Find:** `"ja": "Japanese (日本語)",\n        "en": "English"`

**Replace with:** The full dictionary from Step 1 above.

Remember to test everything thoroughly after making these changes!
