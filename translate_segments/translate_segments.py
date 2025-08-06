import google.generativeai as genai
import os
from collections import defaultdict
from utils import save_cache, read_cache, save_api_key, load_api_key

class SegmentsTranslator:
    def __init__(self,model="gemini-2.0-flash"):
        api_key = load_api_key()
        # If not found, ask user and save it
        if not api_key:
            api_key = input("🔐 Enter your Gemini Api Key: ").strip()
            save_api_key(api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model)

    def translate_batch(self, texts, source_lang="ja", target_lang="en"):
        prompt = f"Translate the following sentences from {source_lang} to {target_lang} which is from an character from an anime or a movie putting in my mind that they could say something figuratively.Keep the same length and meaning. Give me only the translation without explanation nor added text in the begining and don't tell me Here are the translations:\n"
        for i, text in enumerate(texts, 1):
            prompt += f"{i}. {text}\n"

        response = self.model.generate_content(prompt)
        translated = response.text.strip().split('\n')

        # Optional: remove numbering if present in Gemini's response
        translations = []
        for line in translated:
            line = line.strip()
            if line and line[0].isdigit() and '.' in line:
                translations.append(line.split('.', 1)[1].strip())
            else:
                translations.append(line)

        # Pad if output has fewer lines than input (Gemini sometimes skips)
        while len(translations) < len(texts):
            translations.append("")

        return translations

    def translate_segments(self, transcribed_segments, source_lang="ja", target_lang="en", read_from_cache=False, cache_path=None):
        translations = read_cache(read_from_cache, cache_path)
        if translations:
            print(f"Using cached translations from: {cache_path}")
            return translations
        
        translations = defaultdict(list)

        for speaker, segments in transcribed_segments.items():
            texts = [segment["text"] for segment in segments]

            translated_texts = self.translate_batch(texts, source_lang, target_lang)

            for segment, translation in zip(segments, translated_texts):
                segment_copy = segment.copy()
                segment_copy["translation"] = translation
                translations[speaker].append(segment_copy)

        result = dict(translations)
        
        # Save to cache if cache_path is provided
        if cache_path:
            save_cache(cache_path, result)
            print(f"Translations cached to: {cache_path}")

        return result

