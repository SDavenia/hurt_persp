import os
import json
import pandas as pd
from utils.dataset_lexicon_processing import StandardLexicon, StandardDataset, ExampleExtractor
from utils.dataset_lexicon_processing import sanitize_none_string, map_sociodemographics


""" EXPECTED OUTPUT FORMAT:
============================================================
RESULTS DATAFRAME:
============================================================
       lemma      dataset_source                   example_text      label
0       hate  social_media_posts            I hate this weather   negative
1     stupid  social_media_posts      That was a stupid mistake   negative
2     stupid       movie_reviews           Stop being so stupid  offensive
3       ugly       movie_reviews  The movie was ugly and boring   negative
4  worthless  social_media_posts              You are worthless  offensive
"""

lexicon_id = 'hurtlexcore_EN_clean_with_tags'
lexicon_path = 'data/lexicon/hurtlexcore_EN_clean_with_tags_no_duplicates.csv'

# lexicon_id = 'hurtlex_core_EN'
# lexicon_path = 'data/lexicon/hurtlexcore_EN.tsv'


dataset_paths = {
    # 'mhs': 'data/datasets/mhs/data.csv'
    'annWithAttitudes': 'data/datasets/annWithAttitudes/data.csv',
    'voiced': 'data/datasets/voiced/data.csv',
    'sbic': 'data/datasets/sbic/data.csv',
    'popquorn': 'data/datasets/popquorn/data.csv'
}

output_dir = "data/extracted_examples"
match_by = 'lemma_with_pos'  # Options: 'lemma', 'word', 'lemma_with_pos', 'stem'
sociodemographic_mapping_data_path = "data/sociodemographic_mappings.json"

def main():
    ########### Load and preprocess the lexicon ###########
    lexicon = StandardLexicon(lexicon_id=lexicon_id, lexicon_path=lexicon_path)
    lexicon.print_info()   

    ########### Load and preprocess the datasets ###########
    all_output_dfs = []
    for dataset_id, dataset_path in dataset_paths.items():
        print(f"\n{'='*30}\nProcessing dataset: {dataset_id}\n{'='*30}")
        dataset = StandardDataset(dataset_id=dataset_id, dataset_path=dataset_path)
        dataset.print_info()
        # dataset.dataset.to_csv(f'{dataset_id}_processed.csv', index=False)  # Save processed dataset for inspection

        ############ Extract examples from the datasets using the lexicon ###########
        extractor = ExampleExtractor(standard_dataset=dataset, dataset_id=dataset_id, standard_lexicon=lexicon, match_by=match_by)
        results_df = extractor.results_df
        all_output_dfs.append(results_df)
    
    ############ Combine results from all datasets and save to output directory
    final_output_df = pd.concat(all_output_dfs, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)  
    output_df_path = f'{output_dir}/extracted_examples_{match_by}.csv'
    final_output_df.to_csv(output_df_path, index=False)
    print(f"Extracted examples saved to {output_dir}/extracted_examples_{match_by}.csv")

    ############ Postprocess to standardize sociodemographic traits across the various datasets -> TODO: Ideally at some point move all to the dataset class
    df = pd.read_csv(output_df_path)
    with open(sociodemographic_mapping_data_path, 'r') as f:
        sociodemographic_mappings = json.load(f)

    # Sanitize and standardize None 
    df['sociodemographics'] = [sanitize_none_string(x) for x in df['sociodemographics']]
    
    # Map the sociodemographics while also keeping the original column.
    df['sociodemographics_mapped'] = df['sociodemographics'].apply(lambda x: map_sociodemographics(x, sociodemographic_mappings))
    output_df_mapped_path = f"{output_dir}/extracted_examples_{match_by}_mapped.csv"
    df.to_csv(output_df_mapped_path, index=False)
    print(f"Extracted examples with mapped sociodemographics saved to {output_dir}/extracted_examples_{match_by}_mapped.csv")

if __name__ == "__main__":
    main()