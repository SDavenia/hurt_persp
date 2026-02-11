### Hurt Persp

#### 0. Initial setup
Create a `.env` file with HF

The following datasets can be downloaded directly from Huggingface using the `download_datasets.py`:
- POPQUORN
- voiced

For the other datasets:
- SBIC: dowload `SBIC.v2.dev.csv, SBIC.v2.trn.csv, SBIC.v2.tst.csv` from [here](https://maartensap.com/social-bias-frames/) and place them in the `data/datasets/sbic` folder. The `download_datasets.py` script will handle the conversion into a single .csv file that can be used downstream.
- annWithAttitudes: request access to the authors, and place the large-study dataset in `data/datasets/annWithAttitudes/data.csv`.

MISSING:
- MD Leonardelli

#### 1. Download all datasets
```
python download_datasets.py
```
Downloads all publicly available datasets and stores them to `data/datasets/{dataset_id}/data.csv`

#### 2. Extract examples
```
python extract_ecological.py
```
Given the specified lexicon (default uses a cleaned version of Hurtlex-core-EN), and the matching type (e.g. whether the lexicon lemma/word/stem should match with the corresponding ones in the dataset texts, defaults to lemma), it saves a dataframe with columns `text`, `{lemma | lemma_with_pos | word | stem}, label, sociodemographics, dataset_source`, with standardized format for all_datasets.


#### Models of different sizes?
*Consider models that can be run in float16 precision* \
*Consider only models >= 4b and <= 40b \

- **llama3**: unsloth/Llama-3.1-8B-Instruct / unsloth/Llama-3.1-70B -> Would have to run 70B version quantized
- **gemma3**:  unsloth/gemma-3-4b-it, unsloth/gemma-3-12b-it, unsloth/gemma-3-27b-it -> does not support system prompt?
- **qwen3**: unsloth/Qwen3-4B, unsloth/Qwen3-8B, unsloth/Qwen3-14B, unsloth/Qwen3-32B


