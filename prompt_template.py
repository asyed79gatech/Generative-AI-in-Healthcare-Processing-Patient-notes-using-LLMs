def generate_strict_prompt_with_schema(context_text):
    schema_definition = """
    Your task is to extract structured information from the given context.
    The output must strictly follow this JSON schema:
    {
      "ID": "ID of the patient",
      "Name": "Name of the patient",
      "Age": "Age of the patient",
      "Condition": "Any medical conditions and/or diseases",
      "Experience": "Patient experiences",
      "Advice": "Any precautions or advice given to the patient",
      "Prescription": "Medicines suggested or prescribed by the doctor"
    }

    **Rules**:
    1. **Do not include any additional keys or text.**
    2. If any key's value is not explicitly mentioned in the text, leave the value as an empty string.
    3. All seven keys must be present in the output, even if their values are empty.
    4. Only return the JSON object; no explanations or text outside of it.
    """

    few_shot_examples = """
    Example 1:
    Input:
    ID: 568
    Text: D: Good morning, John Doe. I understand you’ve been experiencing some health issues.
    P: Yes, Doctor, I have headaches and dizziness.
    D: Based on your labs, you have been diagnosed with hypertension. Regular exercise and a low-salt diet will help manage it. Additionally, I prescribe Amlodipine 5mg daily.

    Output:
    {
      "ID": "568",
      "Name": "John Doe",
      "Age": "",
      "Condition": "Hypertension",
      "Experience": "Headaches and dizziness",
      "Advice": "Regular exercise and a low-salt diet",
      "Prescription": "Amlodipine 5mg daily"
    }

    Example 2:
    Input:
    ID: 735
    Text: D: Hello Linda Wong. You’re 74 years old, and I see you’re experiencing some issues.
    P: Yes, I’ve been having mood swings and diarrhea.
    D: After examination, I diagnosed you with hyperthyroidism. To manage it, I advise taking regular rest and following up with an endocrinologist.

    Output:
    {
      "ID": "735",
      "Name": "Linda Wong",
      "Age": "74",
      "Condition": "Hyperthyroidism",
      "Experience": "Mood swings and diarrhea",
      "Advice": "Regular rest and follow-up with an endocrinologist",
      "Prescription": ""
    }

    Example 3:
    Input:
    ID: 243
    Text: D: Good evening, Peter. I see you’ve had a follow-up appointment for impetigo.
    P: Yes, Doctor. I still have a skin rash and blisters on my face.
    D: The impetigo persists. I recommend soaking the affected area in warm water thrice daily and using antibiotics. Let me know if symptoms worsen.

    Output:
    {
      "ID": "243",
      "Name": "Peter",
      "Age": "",
      "Condition": "Impetigo",
      "Experience": "Skin rash and blisters on face",
      "Advice": "Soak affected area in warm water thrice daily",
      "Prescription": "Antibiotics"
    }
    """

    prompt = f"""
    {schema_definition}

    {few_shot_examples}

    Now, extract the information for the context below in the exact same JSON format.

    Context:
    {context_text}

    Output only the JSON object. Do not include any other text outside the JSON object.
    """

    return prompt
