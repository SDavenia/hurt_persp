import os
import pandas as pd
from utils.data import StandardLexicon, StandardDataset, ExampleExtractor


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

lexicon_id = 'hurtlex_core_EN_clean'
lexicon_path = 'data/lexicon/hurtlexcore_EN_clean.csv'

# lexicon_id = 'hurtlex_core_EN'
# lexicon_path = 'data/lexicon/hurtlexcore_EN.tsv'


dataset_paths = {
    'mhs': 'data/datasets/mhs/data.csv'
}

output_dir = "data/extracted_examples"
match_by = 'word'

def main():
    ########### Load and preprocess the lexicon ###########
    lexicon = StandardLexicon(lexicon_id=lexicon_id, lexicon_path=lexicon_path)
    lexicon.print_info()
   

    ########### Load and preprocess the datasets ###########
    all_output_dfs = []
    for dataset_id, dataset_path in dataset_paths.items():
        dataset = StandardDataset(dataset_id=dataset_id, dataset_path=dataset_path)
        dataset.print_info()
        # dataset.dataset.to_csv(f'{dataset_id}_processed.csv', index=False)  # Save processed dataset for inspection

        ############ Extract examples from the datasets using the lexicon ###########
        extractor = ExampleExtractor(standard_dataset=dataset, dataset_id=dataset_id, standard_lexicon=lexicon, match_by=match_by)
        results_df = extractor.results_df
        all_output_dfs.append(results_df)
    
    # Combine results from all datasets and save to output directory
    final_output_df = pd.concat(all_output_dfs, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)  
    final_output_df.to_csv(f'{output_dir}/extracted_examples_{match_by}.csv', index=False)
    print(f"Extracted examples saved to {output_dir}/extracted_examples_{match_by}.csv")

if __name__ == "__main__":
    main()