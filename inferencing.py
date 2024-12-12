from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class ModelInferencer:
    def __init__(self, model_name):
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
