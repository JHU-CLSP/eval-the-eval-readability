from datasets import load_dataset
import argparse
import torch
from tqdm import tqdm
import time
import logging
import pandas as pd
import re 
import csv


def main(args):
    if args.subset:
        ds = load_dataset(args.dataset_name, args.subset)
    else:
        ds = load_dataset(args.dataset_name)
    summaries = ds[args.split][args.summary_col]

    with open(args.outfile, 'w') as fout:
        for i in range(len(summaries)):
            if type(summaries[i]) is list:
                summ =  summaries[i][0]
            elif type(summaries[i]) is str:
                summ = summaries[i]
            else:
                raise ValueError
            summaries[i] = summ.replace('\n', ' ').strip()
        logging.info(f"Writing {len(summaries)} to {args.outfile}")
        fout.write('\n'.join(summaries))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--subset')
    parser.add_argument('--split', required=True)
    parser.add_argument('--summary_col', required=True)
    parser.add_argument('--outfile', required=True)
    # parser.add_argument('out_path')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')