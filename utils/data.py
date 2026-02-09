## Code that loads different types of datasets and standardizes them into a common format.
## The common format should have:
## Text column: the text of the post/tweet/comment
## Label column: the label of the post/tweet/comment (need to select the specific dataset label)
## Sociodemographics column


# Dictionary containing text column name, label column name and sociodemographics column name for each dataset.
# text_co
dataset_info = {

}

lexicon_info = {
    
}

class StandardizedDataset:
    def __init__(self, dataset, dataset_id):
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.dataset_info = dataset_info[dataset_id] 
        
        self.preprocess_fns = self.dataset_info['preprocess_fns']

        # Preprocess the dataset (generic):
        self._add_lemma_column()
        self._standardize_labels()

        # Preprocess the dataset (dataset specific):
        # ADD THEM TO DATASET_INFO AS preprocesss_fn = ['function1', 'function2', ...] and then call them here in a loop.
        self._apply_dataset_specific_preprocessing()


    def _lemmatize_text(self, text):
        pass

    def _add_lemma_column(self):
        text_column_name = self.dataset_info['text_column']
        self.dataset['lemma'] = self.dataset[text_column_name].apply(self._lemmatize_text)

    def _standardize_labels(self):
        label_column_name = self.dataset_info['label_column']
        # Add any preprocessing that is dataset specific..
 

    def _apply_dataset_specific_preprocessing(self):
        if self.preprocess_fns:
            pass



class StandardLexicon:
    """
    The standardized lexicon should have the following columns:
    - id: unique identifier for each entry
    - word: the word (without POS)
    - pos: the part of speech of the word (if available)
    - base_lemma: the base lemma of the word (if the word is not a base lemma)
    - is_emoji: whether the word is an emoji or not
    """

    def __init__(self, lexicon, lexicon_id):
        self.lexicon = lexicon
        self.lexicon_id = lexicon_id
    


class WordExampleExtractor:
    """
    Takes as input standardized dataset, dataset id and standardized lexicon.
    Return a pandas dataframe with columns:
     - text: the text of the post/tweet/comment
     - lemma: the lemma of the word in the lexicon that is present in the text
     - label: the label of the post/tweet/comment (need to select the specific dataset label)
     - sociodemographics: the sociodemographics of the post/tweet/comment (as a dict with {'trait1': value1, 'trait2': value2, ...})
     - dataset_source: the dataset id.
    """

    def __init__(self,
                 standardized_dataset: StandardizedDataset,
                 dataset_id: str,
                 lexicon: StandardLexicon):
        self.standardized_dataset = standardized_dataset
        self.dataset_id = dataset_id
        self.lexicon = lexicon
    
    

