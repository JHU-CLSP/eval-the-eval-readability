import pandas as pd
import csv
import argparse
import logging
import time

no_prompt = "{}"

# https://www.ascb.org/science-policy-public-outreach/science-outreach/communication-toolkits/best-practices-in-effective-science-communication/
ascb_prompt = """\
On a scale of  1 to 5, what is the reading ease of the following text? 1 indicates the text requires expert background knowledge and 5 indicates the text is readable to the general population. 
Charateristics of a highly readable text include:
- Know your audience, and focus and organize your information for that particular audience.
- Focus on the big picture. What larger problem is your work a part of? What major ideas or issues does your work address? How will your work help global understanding of some issue?
- Avoid jargon. If you must use a technical term, make sure to explain it, but simplify the language.
- Try to use metaphors or analogies to everyday experiences that people can relate to.
- Underscore the importance of public support for exploratory research and scientific information, and the role of this information in providing the context for effective policy making.

Assume the reader is an adult. Do not use Flesch-Kincaid or other readability formulas. Use your own judgment to rate the text.

Format the output as follows:
Score: <score>
Reason: <reasoning>

Text: {}"""

own_reasoning_prompt = """\
On a scale of  1 to 5, what is the reading ease of the following text? 1 indicates the text requires expert background knowledge and 5 indicates the text is readable to the general population. 
Assume the reader is an adult. Do not use Flesch-Kincaid or other readability formulas. Use your own judgment to rate the text.

Format the output as follows:
Score: <score>
Reason: <reasoning>

Text: {}"""

simple_prompt = """\
On a scale of  1 to 5, what is the reading ease of the following text? 1 indicates the text requires expert background knowledge and 5 indicates the text is readable to the general population. Assume the reader is an adult.

Format the output as follows:
Score: <score>
Reason: <reasoning>

Text: {}"""

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
    

