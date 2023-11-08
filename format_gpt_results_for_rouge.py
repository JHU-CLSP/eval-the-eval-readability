'''
Script to eval accuracy of results
'''
import argparse
import logging
import time
import pandas as pd
import csv
import re
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def main(args):
    df = pd.read_csv(args.input_file)
    df[args.model_name] = df[args.model_name].fillna('')

    with open(args.output_file, 'w') as fout:
        for ex in df[args.model_name]:
            ex = ex.replace('\n', '').strip()
            fout.write(ex + '\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('model_name')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')