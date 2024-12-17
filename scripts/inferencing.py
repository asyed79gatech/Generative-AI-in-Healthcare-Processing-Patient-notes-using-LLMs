from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class ModelInferencer:
    """
    A class for performing inference with a language model on patient notes, designed to handle both single and concurrent processing of notes. 

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for the pre-trained language model.
        model (AutoModelForCausalLM): The pre-trained language model for causal text generation.
        llm (pipeline): The pipeline for text generation using the loaded model and tokenizer.

    Methods:
        process_single_note(note):
            Processes a single patient note by generating a structured JSON response based on a strict prompt.

        process_patient_notes(context, max_workers=8):
            Processes multiple patient notes concurrently, utilizing GPU acceleration and multithreading.
    """

    def __init__(self, model_name):
        """
        Initializes the ModelInferencer class by loading the tokenizer, model, and text generation pipeline.

        Args:
            model_name (str): The name or path of the pre-trained language model to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, max_length=2000, device='cuda')

    def process_single_note(self, note):
        
        prompt = generate_strict_prompt_with_schema(note)
        response = self.llm(prompt, max_length=2000, temperature=0.7, top_p=0.9, pad_token_id=self.tokenizer.pad_token_id)
        generated_text = response[0]["generated_text"]
        extracted_json = generated_text[len(prompt):].strip()
        return extracted_json

    def process_patient_notes(self, context, max_workers=8):
        """
        Processes multiple patient notes concurrently using a thread pool.

        Each note is processed independently by invoking `process_single_note`, allowing for parallel 
        execution and leveraging GPU acceleration.

        Args:
            context (list): A list of patient notes to be processed.
            max_workers (int, optional): The maximum number of threads to use for parallel processing. 
                                         Defaults to 8.

        Returns:
            list: A list of JSON-formatted responses generated for each note.

        Raises:
            Exception: If any error occurs while processing a note, the error is logged, and processing continues.
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_note = {
                executor.submit(self.process_single_note, note): note
                for note in context
            }

            for future in tqdm(as_completed(future_to_note), total=len(context)):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error processing note: {e}")

        return results
