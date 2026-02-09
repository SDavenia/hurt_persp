### Hurt Persp

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


