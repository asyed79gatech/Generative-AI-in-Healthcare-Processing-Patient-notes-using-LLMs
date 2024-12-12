import pandas as pd

class PostProcessor:
    def __init__(self, data_list):
        self.lookup = {d["ID"]: d for d in data_list}

    def get_text(self, row):
        transcript = row["Transcript"]
        question = row["Question"]

        if str(transcript) in self.lookup:
            data = self.lookup[str(transcript)]
            mapping = {
                "What is the patient's name?": data.get("Name", ""),
                "What is the patient's age?": data.get("Age", ""),
                "What is the patient's condition?": data.get("Condition", ""),
                "What symptoms is the patient experiencing?": data.get("Experience", ""),
                "What precautions did the doctor advise?": data.get("Advice", ""),
                "What drug did the doctor prescribe?": data.get("Prescription", ""),
            }
            return mapping.get(question, "")
        return ""

    def post_process(self, test_df):
        test_df["Text"] = test_df.apply(self.get_text, axis=1)
        submission = test_df[["Id", "Text"]]
        return submission
