import os
import json
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

class DataPreprocessor:
    def __init__(self, transcript_path, test_path):
        self.transcript_path = transcript_path
        self.test_path = test_path
        self.context = []

    def load_data(self):
        with open(self.transcript_path, 'r') as json_file:
            self.data = json.load(json_file)

        with open(self.test_path, 'r') as test_file:
            self.test_df = pd.read_csv(test_file)

    def prepare_context(self):
        for k in self.data.keys():
            self.context.append(f"ID: {k}\nText: {self.data[k]}")
        return self.context

    @staticmethod
    def detect_and_translate(transcripts):
        non_english_indices = []

        for i, text in enumerate(transcripts):
            try:
                lang = detect(text)
                if lang != "en":
                    non_english_indices.append(i)
                    translation = translator(text, src_lang=lang, tgt_lang="en")
                    translated_text = translation[0]["translation_text"]
                    transcripts[i] = translated_text
            except Exception as e:
                print(f"Error processing transcript {i}: {e}")

        return transcripts, non_english_indices

    @staticmethod
    def preprocess_text_without_lemmatization_stopwords(input_list):
        def clean_text(text):
            text = re.sub(r"[^\w\s]", " ", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"-", "", text)
            text = text.lower()
            tokens = word_tokenize(text)
            return " ".join(tokens)

        processed_list = []
        for entry in input_list:
            match = re.match(r"(ID:\s*\d+)\s*Text:\s*(.*)", entry, re.DOTALL)
            if match:
                id_part = match.group(1)
                text_part = match.group(2)
                clean_text_part = clean_text(text_part)
                processed_entry = f"{id_part} Text: {clean_text_part}"
                processed_list.append(processed_entry)

        return processed_list
