import argparse
import numpy as np
from tqdm import tqdm
import time
import logging


def main(args):
    texts = open(args.datapath).readlines()
    n_tokens = np.zeros((len(texts,)))
    for i, t in tqdm(enumerate(texts)):
        n = t.strip().split()
        n_tokens[i] = len(n)
    print(f"N docs = {len(texts)}")
    print(f'Min tokens = {np.min(n_tokens)}')
    print(f'Max tokens = {np.max(n_tokens)}')
    print(f'Mean tokens = {np.mean(n_tokens)}')
    print(f'Median tokens = {np.median(n_tokens)}')



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    main(args)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')