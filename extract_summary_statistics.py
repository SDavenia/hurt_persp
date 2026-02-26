import pandas as pd
from utils.dataset_lexicon_processing import StandardDataset

dataset_paths = {
    'annWithAttitudes': 'data/datasets/annWithAttitudes/data.csv',
    'voiced': 'data/datasets/voiced/data.csv',
    'sbic': 'data/datasets/sbic/data.csv',
    'popquorn': 'data/datasets/popquorn/data.csv'
}

extracted_examples_df_path = "data/extracted_examples/extracted_examples_lemma_with_pos_mapped.csv"

def main():
    ########### Load and preprocess the datasets in standardized format ###########
    extracted_examples_df = pd.read_csv(extracted_examples_df_path)
    for dataset_id, dataset_path in dataset_paths.items():
        print(f"\n{'='*30}\nDataset statistics: {dataset_id}\n{'='*30}")
        standard_dataset = StandardDataset(dataset_id=dataset_id, dataset_path=dataset_path, preprocess=False)

        num_annotations = len(standard_dataset.dataset)
        # num_offensive_annotations = sum(standard_dataset.dataset['label'] == 'hurtful')
        # num_non_offensive_annotations = sum(standard_dataset.dataset['label'] == 'not hurtful')
        num_unique_annotators = len(set(standard_dataset.dataset["annotator_id"]))   
        num_unique_texts = len(set(standard_dataset.dataset["text_id"]))   
        num_annotations_per_instance_avg = num_annotations / num_unique_texts
        num_annotations_per_annotator_avg = num_annotations / num_unique_annotators

        # Extract sentences for which at least an annotator labeled as offensive
        offensive_texts = set(standard_dataset.dataset[standard_dataset.dataset['label'] == 'hurtful']['text_id'])

        print(f"Total annotations: {num_annotations}")
        print(f"Unique annotators: {num_unique_annotators}")
        print(f"Unique texts: {num_unique_texts}")
        print(f"Unique offensive texts (inclusive): {len(offensive_texts)} ({round(len(offensive_texts) / num_unique_texts, 2)*100}%)")
        print(f"Num_annotations_per_instance_avg: {round(num_annotations_per_instance_avg, 2)}")
        print(f"num annotations per annotator avg: {round(num_annotations_per_annotator_avg, 2)}")

        # Analyze the extracted examples from the resource
        print(f"{'='*10}Extracted resource statistics{'='*10}")
        extracted_examples_df_dataset = extracted_examples_df[extracted_examples_df["dataset_source"] == dataset_id]
        num_instances_extracted = len(extracted_examples_df_dataset)
        num_unique_texts_extracted = len(set(extracted_examples_df_dataset["text_id"]))
        num_unique_annotators_extracted = len(set(extracted_examples_df_dataset["annotator_id"]))

        # Find what percentage of the texts which were labeled as offensive in the original dataset were extracted in the resource
        num_offensive_texts_extracted = len(set(extracted_examples_df_dataset[extracted_examples_df_dataset['label'] == 'hurtful']['text_id']))

        print(f"Number of text-lemma-annotation triplets extracted: {num_instances_extracted}")
        print(f"Number of unique texts extracted: {num_unique_texts_extracted} (coverage: {round(num_unique_texts_extracted / num_unique_texts, 2)*100}%)")
        print(f"Number of unique offensive texts extracted: {num_offensive_texts_extracted} (coverage: {round(num_offensive_texts_extracted / num_unique_texts_extracted, 2)*100}%)")
        print(f"Number of unique annotators extracted:  {num_unique_annotators_extracted} (coverage: {round(num_unique_annotators_extracted / num_unique_annotators, 2)*100}%)")
        
    

if __name__ == '__main__':
    main()
       
    