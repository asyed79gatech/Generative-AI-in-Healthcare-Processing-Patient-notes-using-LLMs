import os
import json
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

class DataPreprocessor:

    """
    A class for preprocessing and preparing textual data from transcripts for use in LLM-based projects.

    This class provides methods to load data, prepare context, detect non-English text, translate it to 
    English, and preprocess the text by cleaning and tokenizing it.

    Attributes:
        transcript_path (str): Path to the JSON file containing transcripts.
        test_path (str): Path to the CSV file containing test questions and transcript IDs.
        context (list): List to store the prepared context data.

    Methods:
        load_data():
            Loads the transcript JSON file and test CSV file into memory.
        
        prepare_context():
            Prepares a context list where each entry contains an ID and the corresponding text in a formatted string.

        static detect_and_translate(transcripts):
            Detects the language of the transcripts and translates non-English text to English.
            Returns the translated transcripts and indices of non-English entries.

        static preprocess_text_without_lemmatization_stopwords(input_list):
            Cleans and tokenizes text by removing special characters, extra spaces, and converting it to lowercase.
            Maintains the ID and reformats the text while processing.
    
    """
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
