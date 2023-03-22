'''
Script to convert data from Mark to correct data for GPT script
'''
import argparse
import logging
import time
import pandas as pd
import csv

def main(args):
    df = pd.read_csv(args.input_file)
    prompt = open(args.prompt_file).read()

    add_prompt = lambda article: prompt.replace("{ARTICLE_GOES_HERE}", article)

    # Fill NA
    df['text_english'] = df['text_english'].fillna('')

    # Add prompt
    if args.chatgpt_system:
        df['system'] = open(args.chatgpt_system).read()
    
    df["question"] = df['text_english'].map(add_prompt)

    df.to_csv(args.output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('prompt_file')
    parser.add_argument('output_file')
    parser.add_argument('--chatgpt-system')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')