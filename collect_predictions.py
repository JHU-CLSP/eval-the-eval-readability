import argparse
import csv
import json
import openai
import hashlib
import time
import tqdm
import os
import sys
csv.field_size_limit(sys.maxsize)
from openai.error import InvalidRequestError, APIError, RateLimitError

def call_chatgpt_api(args, model_name, input_text, system, stop_tokens):
    messages = []
    if system:
        messages = [{"role": "system", "content": system}]

    messages += [
        {"role": "user", "content": input_text}
    ]
    result = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=args.temperature,
        )

    api_call_parameters = {
        "model": model_name,
        "prompt": input_text,
        "stop": stop_tokens,
        "temperature": args.temperature,
    }
    return result, api_call_parameters

def call_openai_api(args, model_name, input_text, stop_tokens):
    # Call API
    # https://beta.openai.com/docs/api-reference/completions/create?lang=python
    # POST https://api.openai.com/v1/completions
    if not args.use_stop_tokens:
        result = openai.Completion.create(
        model=model_name,
        prompt=input_text,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        logprobs=args.logprobs
        )

        api_call_parameters = {
            "model": model_name,
            "prompt": input_text,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "logprobs": args.logprobs
        }
    else:  # Use stop tokens is True
        result = openai.Completion.create(
        model=model_name,
        prompt=input_text,
        stop=stop_tokens,
        temperature=args.temperature,
        )

        api_call_parameters = {
            "model": model_name,
            "prompt": input_text,
            "stop": stop_tokens,
            "temperature": args.temperature,
        }

    return result, api_call_parameters

def call_api(args, model_name, row, input_text, stop_tokens, max_tries=3):
    text_response = None
    result = {}
    n_tries = 0
    while (text_response is None) and (n_tries < max_tries):
        if model_name == "gpt-3.5-turbo":
            # If there are system instructions
            if 'system' in row:
                system = row['system']
            else:
                system = None
            # Truncate input if we reach the input limit
            try:
                result, api_call_parameters = call_chatgpt_api(args, model_name, input_text, system, stop_tokens)
                text_response = result['choices'][0]['message']['content'].strip()
            except InvalidRequestError:
                input_text = truncate_question(input_text, truncate_input=args.truncate_input)
            except APIError:
                time.sleep(5)
            except RateLimitError:
                time.sleep(60)


        # Other GPT Models
        else:
            try:
                result, api_call_parameters = call_openai_api(args, model_name, input_text, stop_tokens)
                text_response = result['choices'][0]['text'].strip()
            except InvalidRequestError:
                input_text = truncate_question(input_text, truncate_input=args.truncate_input)
            except APIError:
                time.sleep(5)
            except RateLimitError:
                time.sleep(60)

        n_tries += 1
    return result, text_response

def truncate_question(question, truncate_input=2000):
    split_question= question.split(' ')
    split_question = split_question[:truncate_input]
    question = ' '.join(split_question)
    return question

def hash_string(string):
    hash_object = hashlib.sha256(string.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig[:8]
    
def parse_args():
    parser = argparse.ArgumentParser(description="Collecting predictions from the OpenAI API.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the CSV file.",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output CSV file.",
        required=True,
    )
    parser.add_argument(
        "--model-names",
        type=str,
        default=None,
        help="A comma separate list of model names to query.",
        required=True,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="What sampling temperature to use.",
        required=False,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help=(
            "The maximum number of tokens to generate in the GPT-3 completion."
        ),
        required=False,
    )
    parser.add_argument(
        "--truncate-input",
        type=int,
        default=2500,
        help=(
            "The number of tokens to truncate the input to."
        ),
        required=False,
    )
    parser.add_argument(
        "--use-stop-tokens",
        type=bool,
        help=(
            "Whether to use stop tokens (instead of max_tokens) as stopping criteria."
        ),
    )
    parser.add_argument(
        "--logprobs",
        type=bool,
        help=(
            "From OpenAI documentation: Include the log probabilities on the `logprobs` most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response."
        ),
    )
    parser.add_argument(
        "--max-api-tries",
        type=int,
        default=3,
        help=(
            "Maximum number of times to retry an API Call before quitting."
        )
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help=(
            "Example index to resume from. Use when a script crashes and you need to restart."
        )
    )
    args = parser.parse_args()

    return args


# Append prediction dictionary to JSONL file
def append_record(file_path, record):
    with open(file_path, 'a') as f:
        json.dump(record, f)
        f.write(os.linesep)


if __name__ == "__main__":

    openai.api_key = os.environ.get('OPENAI_API_KEY')

    args = parse_args()
    print(args)

    stop_tokens = None
    if args.use_stop_tokens:
        stop_tokens = ['<|endoftext|>']  # Can add additional stop tokens here

    model_names = args.model_names.split(',')
    output_mode = 'a' if args.resume_from > 0 else 'w'

    with open(args.input, 'r', encoding="utf-8") as input, open(args.output, output_mode, newline='') as output:
        reader = csv.DictReader(input)
        writer = None
        
        for ii, row in tqdm.tqdm(enumerate(reader), desc='API'):
            if row['question'] and ii >= args.resume_from:
                question = row['question']
                input_text = question + '\n\n'
                
                if 'hash' not in row:
                    row['hash'] = hash_string(question)

                for model_name in model_names:
                    if ii > 0:
                        time.sleep(1)
                    result, text_response = call_api(args, model_name, row, input_text, stop_tokens, max_tries=args.max_api_tries)
                    row[model_name] = text_response
                    row['result_' + model_name] = json.dumps(result)
            
                if not writer:
                    columns = row.keys()
                    writer = csv.DictWriter(output, fieldnames=columns)
                    if output_mode=='w': # Only write header when writing to a new file 
                        writer.writeheader()
                
                # print(row)
                writer.writerow(row)
                output.flush()
