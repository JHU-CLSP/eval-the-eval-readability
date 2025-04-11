import pandas as pd
import csv
import argparse
import logging
import time

no_prompt = "{}"

simple_prompt = """\
On a scale of  1 to 5, what is the reading ease of the following summary? 1 indicates a very difficult read and 5 indicates a very easy read. 

Summary:
{}"""

demographics_prompt = """\
Given the following person’s demographic information, rate the reading ease of the following summary on a scale of 1 to 5. 1 indicates a very difficult read and 5 indicates a very easy read. 

Demographic Information:
Age: {}
Gender: {}
Education: {}
Involved in research: {}
Number of STEM Courses after high school: {}
Topic familiarity: {} / 5

Summary:
{}"""

topic_familiarity_prompt = """\
Research has found that people with familiarity of a topic have an easier time understanding it and rate those texts as more readable. Texts that have a lot of domain specific details are less readable to people who are not familiar with the topic.
We can rate someone's familiarity with a topic using the following scale:
Use the scale below to rate the reading ease of the given summary.
1 - The text is requires an expert reader  to understand.
2 - A person unfamilar with the topic can understand some of the text.
3 - A person unfamilar with the topic can understand about half of the text.
4 - A person unfamilar with the topic can understand most of the text.
5 - A person unfamiliar with the topic can fully understand the text.

Using this scale, read the following summary. If it contains domain specific details and vocabulary, and the person is unfamiliar with the topic, it should be less readable. If the text has more general terminology, then it would be readable for people who are and who are not familiar with the text.

For a person with {} / 5 familiarity with the topic, what is the reading ease of the following summary?
Summary:
{}"""

simple_prompt_adult = """\
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
    
    


class DemographicsDataProcessor(DataProcessor):
    def __init__(self, data_path, annotator_data_path):
        super(DemographicsDataProcessor, self).__init__(data_path, prompt_name)
        self.prompt_name = prompt_name
        self.prompt = demographics_prompt if prompt_name=="demographics_prompt" else topic_familiarity_prompt
        self.annotator_data = pd.read_csv(annotator_data_path)
        self.summaries = self.read_summaries()

    def read_summaries(self):
        summ_df = pd.read_csv(self.data_path)
        summaries = {}
        for _, row in summ_df.iterrows():
            summaries[(row['name'], row['textVersion'], row['studyNum'])] = row['cleanText']
        return summaries
    
    def format_prompt(self, row):
        paper_id = (row['paperId'], row['complexityLevel'], row['study'])
        if self.prompt_name == "demographics_prompt":
            return self.prompt.format(
                    row['age'],
                    row['gender'],
                    row['education'],
                    row['researchInvolved'],
                    row['stemEducation'],
                    row['article_familiar'],
                    self.summaries[paper_id]
                )
        return self.prompt.format(
                    row['article_familiar'],
                    self.summaries[paper_id]
                )

    def get_messages(self):
        summaries = pd.read_csv(self.data_path)
        messages = []
        for _, row in self.annotator_data.iterrows():
            if row['complexityLevel'] != 'original':
                prompt = self.format_prompt(row)
                messages.append([{"role": "user", "content": prompt}])

        return messages
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cleaned_summaries')
    parser.add_argument('annotator_data')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    data = DemographicsDataProcessor(args.cleaned_summaries, args.annotator_data)
    messages = data.clean_data()
    print(messages[0])
    print(len(messages))
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')
