from data_preprocessing import DataPreprocessor
from inferencing import ModelInferencer
from post_process import PostProcessor

if __name__ == "__main__":
    # File paths
    transcript_path = "transcripts.json"
    test_path = "test.csv"
    save_path = "submission.csv"

    # Preprocessing
    preprocessor = DataPreprocessor(transcript_path, test_path)
    preprocessor.load_data()
    context = preprocessor.prepare_context()
    context, _ = preprocessor.detect_and_translate(context)
    processed_context = preprocessor.preprocess_text_without_lemmatization_stopwords(context)

    # Inferencing
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    inferencer = ModelInferencer(model_name)
    results = inferencer.process_patient_notes(processed_context)

    # Post-processing
    post_processor = PostProcessor(results)
    submission_df = post_processor.post_process(preprocessor.test_df)
    submission_df.to_csv(save_path, index=False)

    print(f"Submission file saved to {save_path}")
