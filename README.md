# Extracting Structured Information from Unstructured Patient Notes using Open-Source LLMs
In healthcare, accurate and efficient documentation is critical, but it can be a time-consuming process for medical professionals. This project addresses the challenge of automating the extraction of structured data from unstructured patient notes (patient-doctor conversation transcripts) using open-source Large Language Models (LLMs). By doing so, healthcare workers can focus more on patient care and less on administrative tasks.

The project was inspired by the 2023 Purdue "Data for Good" competition, which aimed to bridge healthcare needs with cutting-edge AI solutions. This system leverages open-source LLMs to process medical conversations and extract the necessary information to populate forms or databases.

![Masthead](/Patinet_Notes_Processing.png)
## Motivation
This project showcases the potential of open-source LLMs, prompt engineering, and data pre-processing to automate tasks for healthcare professionals. It gives an insight into imagining a world where healthcare workers and volunteers can focus entirely on patient care, free from the burden of time-consuming paperwork. It was particularly interesting because it is a prime example of how AI simplifies administrative processes, ensuring data accuracy, consistency, and security, while freeing up valuable time and resources.

## Technology Used
**1- Python:** For data collection, preprocessing, and integration with LLMs.

**2- Open-Source LLMs:**
- facebook/m2m100_418M for translation of non-enlgish patient notes into english as part of pre-processing

- Mistral-7B-Instruct-v0.3 for extracting structured information from patient note by providing the note in the prompt as context

**3- Libraries:**

- `pandas` and `numpy` for data manipulation.

- `transformers` for model integration and creating LLM pipelines.

- `spacy`, `NLTK` and `Regex`  for linguistic preprocessing of notes.

- `json` for managing input/output data formats.

- `langdetect` for patient note language translation pipeline

- `concurrent.futures` module of the python's standard library to run LLM inferencing tasks on GPU asynchronously using the `ThreadPoolExecutor`


**4- Hardware:** Used `NVIDIA A100 GPU` purchased through Google Colab subscription to load the model and run the LLM inferencing.

## Dataset

During an appointment, the conversation between the Patinet and Doctor is recorded as patient notes. These notes/ transcripts form a gold mine, however, being raw data, they end up being both challenging and time-consuming to retrieve specific and targeted information from.

The dataset used in this project was provided by Prediction Guard during the Data 4 Good challenge organized in Fall 2023.

The `transcripts.json` file contains a total of 2001 patient notes identified through the transcript IDs. The following is an example of how patient notes are contained in the `transcripts.json` file as json objects.

```json

{"2055": "During the visit, I examined Mr. Don Hicks, who is 81 years old and presented with a fungal infection. He had dischromic patches, nodal skin eruptions, and skin rash as symptoms. Upon examination, I confirmed the diagnosis of fungal infection. I advised Mr. Hicks to take precautions such as bathing twice a day, using detol or neem in the bathing water, keeping the infected area dry, and using clean cloths. I did not prescribe any medication for him.",

"291": "During the visit, I examined Tina Will, a 69-year-old patient who presented with symptoms of chest pain, vomiting, and breathlessness. After conducting a thorough examination, I determined that Tina was suffering from a heart attack. As a result, I advised her to seek immediate medical attention. Since there were no precautions that could be taken to prevent a heart attack, I did not prescribe any medication. Instead, I recommended that Tina follow up with her primary care physician for ongoing treatment and management of her condition."}

```

For the full dataset, see [transcripts.json](transcripts.json)

In addition to the above `transcripts.json` file, a `test.csv` file was also provided containing tabular form of six structured questions that need to answered for each patient ID corresponding to the ID in the `transcripts.json` file. A glimpse of the csv file is given below:

## Data Overview

Here's a summary of our data:

| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |

For the full dataset, see [test.csv](test.csv)



## Methodology

## Setup Instructions
Follow these steps to run the Project
**1. Clone the Respository**


**2. Prepare and Launch the Envionment**

**3. Generate/Update the data**

**4. Create an HTTP Port to view the network graph**

**5. Explore the network**

## Project Structure
```bash

```
## How to Use


## Future Enhancements


## Acknowledgements



