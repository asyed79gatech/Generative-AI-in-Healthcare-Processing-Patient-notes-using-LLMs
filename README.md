# Extracting Structured Information from Unstructured Patient Notes using Open-Source LLMs
In healthcare, accurate and efficient documentation is critical, but it can be a time-consuming process for medical professionals. This project addresses the challenge of automating the extraction of structured data from unstructured patient notes (patient-doctor conversation transcripts) using open-source Large Language Models (LLMs). By doing so, healthcare workers can focus more on patient care and less on administrative tasks.

The project was inspired by the 2023 Purdue "Data for Good" competition, which aimed to bridge healthcare needs with cutting-edge AI solutions. This system leverages open-source LLMs to process medical conversations and extract the necessary information to populate forms or databases.

![Masthead](/Patinet_Notes_Processing.png)


## Motivation
This project showcases the potential of open-source LLMs, prompt engineering, and data pre-processing to automate tasks for healthcare professionals. It gives an insight into imagining a world where healthcare workers and volunteers can focus entirely on patient care, free from the burden of time-consuming paperwork. It was particularly interesting because it is a prime example of how AI simplifies administrative processes, ensuring data accuracy, consistency, and security, while freeing up valuable time and resources.


## Technology Used
**1- Python:** For data collection, preprocessing, and integration with LLMs.

**2- Open-Source LLMs:**
- `facebook/m2m100_418M` for translation of non-enlgish patient notes into english as part of pre-processing

- `Mistral-7B-Instruct-v0.3` for extracting structured information from patient note by providing the note in the prompt as context

**3- Libraries:**

- `pandas` and `numpy` for data manipulation.

- `transformers` for model integration and creating LLM pipelines.

- `spacy`, `NLTK` and `Regex`  for linguistic preprocessing of notes.

- `json` for managing input/output data formats.

- `langdetect` for patient note language translation pipeline

- `concurrent.futures` module of the python's standard library to run LLM inferencing tasks on GPU asynchronously using the `ThreadPoolExecutor`


**4- Hardware:** Used `NVIDIA A100 GPU` purchased through Google Colab subscription to load the model and run the LLM inferencing.


## Dataset Overview

Patient notes recorded during doctor appointments form a valuable resource for extracting insights. However, due to their raw, unstructured nature, retrieving specific, actionable information can be both challenging and time-consuming.

This project utilizes a dataset provided by Prediction Guard during the Data 4 Good Challenge (Fall 2023) to address this challenge. 

#### Dataset Description
Patient notes recorded during doctor appointments form a valuable resource for extracting insights. However, due to their raw, unstructured nature, retrieving specific, actionable information can be both challenging and time-consuming.

This project utilizes a dataset provided by Prediction Guard during the Data 4 Good Challenge (Fall 2023) to address this challenge.

Dataset Description
**1. `transcripts.json`**
Contains 2001 patient notes represented as JSON objects, each identified by a unique transcript ID.
Example of a patient note:


```json

{"2055": "During the visit, I examined Mr. Don Hicks, who is 81 years old and presented with a fungal infection. He had dischromic patches, nodal skin eruptions, and skin rash as symptoms. Upon examination, I confirmed the diagnosis of fungal infection. I advised Mr. Hicks to take precautions such as bathing twice a day, using detol or neem in the bathing water, keeping the infected area dry, and using clean cloths. I did not prescribe any medication for him.",

"291": "During the visit, I examined Tina Will, a 69-year-old patient who presented with symptoms of chest pain, vomiting, and breathlessness. After conducting a thorough examination, I determined that Tina was suffering from a heart attack. As a result, I advised her to seek immediate medical attention. Since there were no precautions that could be taken to prevent a heart attack, I did not prescribe any medication. Instead, I recommended that Tina follow up with her primary care physician for ongoing treatment and management of her condition."}

```

For the complete file, refer to [transcripts.json](transcripts.json)

**2. `test.csv`**
A structured tabular dataset with six questions for each patient note identified by its corresponding transcript ID.
Example rows for Transcript ID: 2055:

| Id | Transcript | Question |
|---|------------|----------|
| 587d0feb-5780-43e1-9595-e19d4b31dc07 | 2055 | What is the patient's name? |
| 263e8884-e8ba-4266-bb0c-85271419a0b3 | 2055 | What is the patient's age? |
| 74c68eca-61b2-49d0-9b1c-0f6f886b04ff | 2055 | What is the patient's condition? |
| 8572ab5d-f20a-4de5-ab44-f42b07e45a00 | 2055 | What symptoms is the patient experiencing? |
| f5c92075-ef05-4fbf-a7a0-aa86c586ff02 | 2055 | What precautions did the doctor advise? |
| 03406fb0-e67d-4614-a745-ed02c7ac6c46 | 2055 | What drug did the doctor prescribe? |

For the complete file, refer to [test.json](test.csv)

## Objective
The task is to extract information from relevant patient notes from the `transcripts.json` file using the transcript ID and generate answers to the six questions listed in the `test.csv` file.

This project aims to streamline the extraction process and transform unstructured notes into structured, actionable insights using open-source LLMs and robust prompt-engineering.

## Methodology
**1- Data Preprocessing**

The `DataPreprocessor` class in data_preprocessing.py handles all preprocessing tasks to prepare the transcripts as meaningful context for the LLM. This ensures accurate and efficient answering of the six patient-related questions.

**Loading Transcripts**

All transcripts from `transcripts.json` are loaded into a Python list called `context`.

**Translating Non-English Transcripts**

Some transcripts are in languages other than English. These are detected and translated using the `detect_and_translate` method:

```python
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
```
An example of the transcript before and after the translation is given below:
*Before*
```
ID: 3538
Text: D: शुभ प्रभात, थॉमस जी। आज मुझे आपकी कैसे सहायता कर सकते हैं? 

P: शुभ प्रभात, डॉक्टर। मुझे अपने लम्बरों में कमजोरी, चक्कर आना और गले में दर्द के बारे में अनेक दिनों से तक समस्या है। 

D: महसूस हो रहा है, थॉमस जी। आपके लक्षणों और आपकी उम्र के आधार पर, मैं आशा कर रहा हूं कि आपको सर्विकल स्पॉन्डिलोसिस हो सकती है। 

P: यह क्या है, डॉक्टर? 

D: सर्विकल स्पॉन्डिलोसिस एक ऐसा स्थिति है जो आपके गले के हिस्से को प्रभावित करता है, जो आपके गले का हिस्सा है। यह 60 साल से ऊपर के लोगों में आम है और यह आपके गले में दर्द और स्टिफ़नस की वजह से हो सकता है, जैसे ही आपके लम्बरों में कमजोरी और नम्बर्स का महसूस हो सकता है। 

P: ओह, मैं समझ गया। क्या मुझे अच्छे से महसूस करने के लिए
```
*After*
```
ID: 3538 

Text: D: Happy Happiness, Thomas G. How can I help you today? 

P: Happy Happiness, Doctor. I have a problem for many days about weakness in my limbs, swelling and throat pain. 

D: Feeling, Thomas G. Depending on your symptoms and your age, I hope you may have generic spondylosis. 

P: What is it, Doctor? 

D: Generic spondylosis is a condition that affects the part of your throat, which is part of your throat. It is common in people over 60 years and it can be caused by the pain and stiffiness in your throat, as well as the weakness and weakness in your limbs.
```

**Cleaning and Tokenizing Transcripts**
The `preprocess_text_without_lemmatization_stopwords` method cleans and tokenizes the transcripts by:
Removing special characters and extra spaces
Converting text to lowercase


```python
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
```


The result is a cleaned, English-only list of tokenized transcripts ready for analysis.

Key Improvements:

- Original Transcripts: May include non-English text, special characters, and inconsistent formatting.
- Processed Transcripts: Consistent formatting, tokenized, cleaned of punctuation, and all in English.


**2- Prompt Engineering**

Prompt engineering plays a critical role in extracting structured, specific information from open-source LLMs. For this project, prompts are designed to ensure outputs strictly follow a predefined JSON format. This makes the extracted information easy to parse and aligns directly with the six questions in the `test.csv` file.

**JSON Schema and Rules**

The LLM is instructed to generate responses in the following JSON format:

```json
{
  "ID": "ID of the patient",
  "Name": "Name of the patient",
  "Age": "Age of the patient",
  "Condition": "Any medical conditions and/or diseases",
  "Experience": "Patient experiences",
  "Advice": "Any precautions or advice given to the patient",
  "Prescription": "Medicines suggested or prescribed by the doctor"
}
```
Rules to Ensure Consistency:

All seven keys must be present, even if their values are empty.
If a key’s value is not mentioned in the text, it should remain an empty string.
The output must be a JSON object, with no additional text or explanations.

**Few-Shot Prompting**

To enhance LLM performance, few-shot prompting is used. This involves providing the model with a few examples of inputs and expected outputs within the prompt.

Why Use Few-Shot Prompting?
Few-shot prompting demonstrates the expected output format and response style, guiding the LLM to generate more consistent and accurate results. By including a few examples of patient notes with their corresponding JSON outputs, the LLM learns:

The correct structure of the response
- How to extract relevant information
- How to handle missing or incomplete data
- Example Prompt with Few-Shot Examples
Here’s a simplified example of how few-shot prompting is implemented:

**Example - 1**

*Input*
```mathematica
ID: 568  
Text: D: Good morning, John Doe. I understand you’ve been experiencing some health issues.  
P: Yes, Doctor, I have headaches and dizziness.  
D: Based on your labs, you have been diagnosed with hypertension. Regular exercise and a low-salt diet will help manage it. Additionally, I prescribe Amlodipine 5mg daily.  

```
*Output:*
```json
{
  "ID": "568",
  "Name": "John Doe",
  "Age": "",
  "Condition": "Hypertension",
  "Experience": "Headaches and dizziness",
  "Advice": "Regular exercise and a low-salt diet",
  "Prescription": "Amlodipine 5mg daily"
}
```
Including three such examples in the prompt helps the model better understand the task. Few-shot prompting ensures the model generates responses in the desired format while minimizing the need for extensive post-processing.

For the full prompt refer to [prompt_template.py](prompt_template.py)


**3- LLM Inferencing**

For inference, the Huggingface `transformers` library is used to load the `Mistral-7B-Instruct-v0.3` model, which supports a 32,000-token context window—more than enough for the prompt and patient notes combined. This eliminates the need for creating vector embeddings of the notes.

To efficiently process 2,001 patient notes on `A100 GPUs`, multithreading is implemented using Python's `concurrent.futures` module. The `ThreadPoolExecutor` class handles parallelism by creating a pool of 8 threads to process notes concurrently. Tasks are submitted via the `submit()` method and retrieved with `future.result()` once completed. 

To speed up the processing of 2001 notes using the `A100` GPUs, the `concurrnt.futures' module is used and multithreading is handled using the ThreadPoolExecutor class. This class allows you to create a pool of threads and execute tasks concurrently.

**Key Methods:**

- `process_single_note(note)`
Processes one patient note using the LLM, structured prompt, and tokenizer. It generates a JSON response adhering to the required schema.
Example code:

```python
def process_single_note(self, note):
    
    prompt = generate_strict_prompt_with_schema(note)
    response = self.llm(prompt, max_length=2000, temperature=0.7, top_p=0.9, pad_token_id=self.tokenizer.pad_token_id)
    generated_text = response[0]["generated_text"]
    extracted_json = generated_text[len(prompt):].strip()
    return extracted_json
```

- `process_patient_notes(context, max_workers=8)`
Processes all notes in parallel, returning a list of dictionaries containing answers to the six questions in `test.csv`.
Example output:

```python
[{
      "ID": "291",
      "Name": "Tina Will",
      "Age": "69",
      "Condition": "Heart attack",
      "Experience": "Chest pain, vomiting, and breathlessness",
      "Advice": "Seek immediate medical attention",
      "Prescription": ""
    }
{
      "ID": "3538",
      "Name": "Thomas",
      "Age": "",
      "Condition": "Sarvical Spondylosis",
      "Experience": "Dizziness, neck pain, and pain in the neck",
      "Advice": "",
      "Prescription": ""
    }]
```

This output is post-processed to populate the corresponding rows in the `test.csv` file. The parallel processing ensures faster and efficient inference for large datasets.


**4- Post-Processing**

After LLM inferencing, the generated `JSON` responses are parsed and mapped to their respective rows in the `test.csv` file. A new `.csv` file is created, maintaining the original structure of the `test.csv` file, with the `ID` column containing unique question IDs and the `Text` column populated with the corresponding LLM responses. This final `.csv` file serves as the output.

Below is an example of the post-processed `test.csv` file:

### Example `test.csv` File

| ID                                   | Text                                                                                               |
|--------------------------------------|----------------------------------------------------------------------------------------------------|
| 587d0feb-5780-43e1-9595-e19d4b31dc07 | Don Hicks                                                                                         |
| 263e8884-e8ba-4266-bb0c-85271419a0b3 | 81                                                                                                |
| 74c68eca-61b2-49d0-9b1c-0f6f886b04ff | Fungal infection                                                                                  |
| 8572ab5d-f20a-4de5-ab44-f42b07e45a00 | Dischromic patches, nodal skin eruptions, and skin rash                                           |
| f5c92075-ef05-4fbf-a7a0-aa86c586ff02 | Bathing twice a day, using Dettol or neem in the bathing water, keeping the infected area dry, and using clean cloths |
| 03406fb0-e67d-4614-a745-ed02c7ac6c46 |                                                                                                    |

for the final file, refer to [first_submission.csv](first_submission.csv)

