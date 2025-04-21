# Evaluating the Evaluators

## Setup


## Datasets
We use the following summarization datasets:
- [arXiv](https://huggingface.co/datasets/armanc/scientific_papers)
- [PubMed](https://huggingface.co/datasets/armanc/scientific_papers)
- [SciTLDR](https://huggingface.co/datasets/allenai/scitldr)
- [Science Journal for Kids](https://huggingface.co/datasets/loukritia/science-journal-for-kids-data)
- [CDSR](https://github.com/qiuweipku/Plain_language_summarization)
- [PLOS](https://huggingface.co/datasets/tomasg25/scientific_lay_summarisation)
- [eLife](https://huggingface.co/datasets/tomasg25/scientific_lay_summarisation)**
- [Eureka](https://github.com/slab-itu/HTSS/)
- [CELLS](https://github.com/LinguisticAnomalies/pls_retrieval)
- [SciNews](https://huggingface.co/datasets/dongqi-me/SciNews)


Our human annotated readability data is from [August et al 2024.](https://dl.acm.org/doi/10.1145/3613904.3642289)

**Data format:** The code expects the data in a text file, with each new line containing a sumamry. For the HuggingFace datasets, use the following command to load and format the data:
```{bash}
$ python scripts/format_data.py \
--dataset_name <HF_DATASET_NAME> \
--subset <DATA_SUBSET> \
--split <DATA_SPLIT> \
--summary_col <COL_NAME_WITH_SUMMARIES> \
--outfile </PATH/TO/SAVE/FORMATTED/DATA>
```

<sub>** This is the official, released version of this dataset. We found multiple grammatical errors and re-collected the dataset for this paper. We are currently working with the original authors of the SJK paper to re-release the cleaned data.</sub>

## Models
We use the following language models:
- [OLMo-2 7B Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)
- [OLMo-2 13B Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct)
- [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Mixtral-8x7B Instruct](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [gemma 1.1 7b Instruct](https://huggingface.co/google/gemma-1.1-7b-it)
- [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)





## Running Code


The following command prompts the model to rate the readability of the summaries in the input file:
```{bash}
$ python main.py \
<MODEL> \
</PATH/TO/INPUT_FILE> \
</PATH/TO/OUTPUT_FILE> \
-bsz <BATCH_SIZE>
```

The following command runs a script to extract the model ratings:
```{bash}
$ python scripts/get_rating.py </PATH/TO/OUTPUT_FILE>
```