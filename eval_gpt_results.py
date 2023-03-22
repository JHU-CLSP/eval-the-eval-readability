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

def clean_pred(pred):
    pred = re.sub(r'[^\w\s]', '', pred) # Remove punctuation
    pred = pred.strip().lower() # strup and lower case

    return pred 

def clean_labels(label):
    # Map labels in data to labels in prompt
    standardization_mapping = {
        "entertainment": "entertainment",
        "agriculture": "agriculture",
        "criminal_justice": "criminal justice",
        "packaging": "packaging",
        "environment": "environment",
        "nicotine": "nicotine",
        "Uncategorized": "none of the above",
        "air": "air",
        "Tobacco Warnings": "health warnings",
        "Tobacco Industry": "industry",
        "Tobacco Bans": "bans (other)",
        "Tobacco Prevalence": "prevalence",
        "Tobacco Prices": "prices",
        "Tobacco Flavor Restrictions": "flavors",
        "Tobacco Cessation": "quitting",
        "Tobacco Advertising": "advertising",
        "Tobacco Products": "none of the above",
        "Tobacco Age Restrictions": "age"
    }

    return standardization_mapping[label]

def main(args):
    df = pd.read_csv(args.input_file)
    df[args.model_name] = df[args.model_name].fillna('')
    # Labels from prompt
    labels = ['advertising', 'age', 'bans (other)', 'flavors', 'industry', 
                'prevalence', 'prices', 'quitting', 'health warnings',
                'criminal justice', 'entertainment', 'air', 'packaging',
                'agriculture', 'environment', 'nicotine', 'screening',
                'tobacco control funding and resources', 'none of the above'
                ]
    invalid_pred_idx = len(labels) 
    labels2idx = {l:i for i,l in enumerate(labels)}

    y_true = map(clean_labels, df['data_to_tw_label'])
    y_true_idx = [labels2idx[y] for y in y_true]
    
    y_pred = map(clean_pred,  df[args.model_name])
    y_pred_idx = [labels2idx[y] if y in labels2idx else invalid_pred_idx 
                  for y in y_pred]
    
    # Save incorrect predictions
    df['y_pred_clean'] = y_pred_idx
    df['y_true_clean'] = y_true_idx
    incorrect = df[df['y_pred_clean']!=df['y_true_clean']]
    print(f'{len(incorrect)}/{len(df)} incorrect articles')
    incorrect.to_csv(args.output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Run evaluation
    print('Accuracy =',accuracy_score(y_true_idx, y_pred_idx))
    print(classification_report(y_true_idx, y_pred_idx, 
                                labels=list(range(len(labels)+1)),
                                target_names=labels + ['invalid response']))

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