# request-gpt

Code to make API calls to OpenAI models. 

## Requirements
To install requirements, run `pip install -r requirements.txt`.

Set an environment variable with your API key:
`$ export OPENAI_API_KEY="{YOUR_KEY_HERE}"`

## Call GPT APIs
The script expects the input file to be a csv with a column `question` for the prompt. 
Optionally, the file may include a `system` column for the system instructions when calling the ChatGPT API.

```
$ python collect_predictions.py \
  --input INPUT        
  --output OUTPUT       
  --model-names MODEL_NAMES
  --temperature TEMPERATURE
  --max-tokens MAX_TOKENS
  --truncate-input TRUNCATE_INPUT
  --use-stop-tokens USE_STOP_TOKENS
  --logprobs LOGPROBS
```

## Tobacco Watcher Scripts
`eval_gpt_results.py` and `to_format_for_gpt.py` are specific scripts for formatting Tobacco Watcher data. 
They could be extended to work for other datasets but have not yet.
