import transformers
import argparse
import torch
from tqdm import tqdm
import time
import logging
from data import get_data_processor
from transformers import AutoModelForCausalLM, AutoTokenizer



def call_model(prompt_batch, max_tokens, terminators, temperature):
    start_time = time.time()
    outputs = pipeline(
        prompt_batch,
        max_new_tokens=max_tokens,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        eos_token_id=terminators,
        temperature=temperature,
        num_return_sequences=1,
    )
    end_time = time.time()
    
    response_batch = [output[0]['generated_text'][len(prompt):] for output, prompt in zip(outputs, prompt_batch)]

    return response_batch

def prompt_os(messages_list, pipeline, temperature, max_tokens, batch_size) -> list:

    response_list = []
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    for i in tqdm(range(0, len(messages_list), batch_size)):
        
        print(f"Running batch {i} to {i + min(batch_size, len(messages_list)-i)} with temperature {temperature}")
        
        messages_batch = messages_list[i : i + batch_size]
        response_list += call_model(messages_batch, max_tokens, terminators, temperature)

    return response_list

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('data_path')
    parser.add_argument('out_path')
    parser.add_argument('-t', '--temperature', dest='temperature', type=float, default=0.8)
    parser.add_argument('-bsz', '--batch_size', dest='batch_size', type=int, default=1)
    parser.add_argument('--max_n', type=int, default=-1)
    parser.add_argument('--prompt_name', default="own_reasoning_prompt")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    encoding = AutoTokenizer.from_pretrained(args.model_name)
    encoding.pad_token = encoding.eos_token
    max_tokens = encoding.model_max_length

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_name,
        device_map="auto", 
        model_kwargs={"torch_dtype": torch.bfloat16},
        batch_size=args.batch_size,
        trust_remote_code=True
    )

    pipeline.tokenizer.pad_token_id = pipeline.model.config.bos_token_id
    pipeline.tokenizer.padding_side = "left"
    logging.info(f"Device map = {pipeline.model.hf_device_map}")

    start = time.time()
    data_processor = get_data_processor(args.data_path, prompt_name=args.prompt_name)
    messages_list = data_processor.get_messages()
    if args.max_n > 0:
        messages_list = messages_list[:args.max_n]
    logging.info(f'Running {len(messages_list)} prompts')
    response_list = prompt_os(messages_list, pipeline, args.temperature, max_tokens, args.batch_size)
    logging.info(f'Saving responses to {args.out_path}')
    data_processor.save_responses(messages_list, response_list, args.out_path)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')