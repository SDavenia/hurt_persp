### Hurt Persp
Repository for the paper "HurtLens: A Perspectivist Corpus Analysis of Hurtful Language".

#### 0. Initial setup
Create a `.env` file with HF valid key.

The following datasets can be downloaded directly from Huggingface using the `prepare_datasets.py`:
- POPQUORN
- voiced: request access via HF beforehand.

For the other datasets:
- SBIC: dowload `SBIC.v2.dev.csv, SBIC.v2.trn.csv, SBIC.v2.tst.csv` from [here](https://maartensap.com/social-bias-frames/) and place them in the `data/datasets/sbic` folder. The `prepare_datasets_lexicon_data.py` script will handle the conversion into a single .csv file that can be used downstream.
- annWithAttitudes: request access to the authors, and place the large-study dataset in `data/datasets/annWithAttitudes/data.csv`.

The lexicon considered for the analysis is included in the repository under `data/lexicon/hurtlexcore_EN_clean_with_tags_no_duplicates_corrected.csv`.

#### 1. Download and Preprocess datasets.
```
python prepare_datasets_lexicon_data.py
```
Downloads all publicly available datasets and stores them to `data/datasets/{dataset_id}/data.csv`

#### 2. Extract examples
```
python extract_ecological.py
```
Given the specified lexicon (default uses a cleaned version of Hurtlex-core-EN), and the matching type (e.g. whether the lexicon lemma/word/stem should match with the corresponding ones in the dataset texts, defaults to lemma, defaults to `lemma_with_pos`), it saves a dataframe with columns `text,{lemma | lemma_with_pos | word | stem},label,sociodemographics,dataset_source,annotator_id,text_id,original_label,tag,sociodemographics_mapped`, with standardized format for all_datasets. \
Additionally, it handles duplicates in text-annotator pairs, removing instances where inconsistencies occur.

#### 3. Extract dataset statistics 
```
python extract_summary_statistics.py
```
Prints to output the statistics of the original datasets and of the corpus consisting of text-lemma-annotator triplets, extracted with the specified lexicon.

#### 4. Exploratory analysis into influence of sociodemographic on ratings of texts containing certain types of offensive words
All the code for visualizations is included in `obtain_plots_text_pairwise.ipynb`.

#### 5. GLMMs analysis
All code to fit the models and to obtain the predicted proabilities for the analysis in the paper is contained in `sociodemographic_tag_analysis.R`. \
For the visualizations, use the code in `sociodemographic_tag_plots.R`.

