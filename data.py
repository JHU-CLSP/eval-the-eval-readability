import pandas as pd
import csv
import argparse
import logging
import time


simple_prompt = """\
On a scale of  1 to 5, what is the reading ease of the following text? 1 indicates the text requires expert background knowledge and 5 indicates the text is readable to the general population. Assume the reader is an adult.

Text:
{}"""

def get_data_processor(data_path, 
                        annotator_data_path=None, 
                        prompt_name="simple_prompt",
                        ):
    if annotator_data_path is None:
        if data_path.endswith('.csv'):
            return DataProcessorCSV(data_path, prompt_name)    
        return DataProcessor(data_path, prompt_name)
    return DemographicsDataProcessor(data_path, annotator_data_path, prompt_name)

class DataProcessor:
    def __init__(self, data_path, prompt_name):
        self.data_path = data_path

        self.prompt = globals()[prompt_name]
    
    def get_messages(self):
        summaries = open(self.data_path).readlines()
        messages = [
            [{"role": "user", "content": self.prompt.format(s.strip())}]
            for s in summaries
        ]

        return messages

    def save_responses(self, prompts, responses, outpath):
        df = pd.DataFrame({
            "prompt": prompts,
            "response": responses
            })
        df.to_csv(outpath, index=False, quoting=csv.QUOTE_ALL)

class DataProcessorCSV(DataProcessor):
    def __init__(self, data_path, prompt_name):
        super(DataProcessorCSV, self).__init__(data_path, prompt_name)

    def get_messages(self):
        df = pd.read_csv(self.data_path)
        messages = df['cleanText'].apply(
            lambda r: [{"role": "user", "content": self.prompt.format(r)}]
        ).tolist()

        return messages
    

