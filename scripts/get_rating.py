import argparse
import torch
from tqdm import tqdm
import time
import logging
import pandas as pd
import re 
import csv

def find_rating(text):
    p = re.compile("reading ease (\D)+((1|2|3|4|5)(\.\d)?)")
    tmp = p.search(text)
    if tmp is not None:
        return tmp.group(2)
    d = re.compile("((1|2|3|4|5)(\.\d)?)")
    tmp2 = d.search(text)
    if tmp2 is not None:
        return tmp2.group(1)
    return -1

def analyze_ratings(df):
    print()
    print(df['reading_ease'].value_counts())


def main(args):
    df = pd.read_csv(args.rating_path)
    ratings = df['response'].apply(find_rating)
    df['reading_ease'] = ratings
    df.to_csv(args.rating_path, index=False, quoting=csv.QUOTE_ALL)
    analyze_ratings(df)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rating_path')
    # parser.add_argument('out_path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')