## Code that loads different types of datasets and standardizes them into a common format.
## The common format should have:
## Text column: the text of the post/tweet/comment
## Label column: the label of the post/tweet/comment (need to select the specific dataset label)
## Sociodemographics column

import nltk
import emoji
import pandas as pd

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

# Dictionary containing text column name, label column name and sociodemographics column name for each dataset.
dataset_info = {

}

lexicon_info = {
    'hurtlex_core_EN': {
            'load_params': {
                'sep': '\t',
                'header': None,
                'names': ['id', 'lemma']
            },
            'preprocess_fn': 'hurtlex_preprocess'
        }
    }


class StandardizedDataset:
    def __init__(self, dataset_id, dataset_path):
        self.dataset_id = dataset_id
        self.dataset_path = dataset_path
        self.dataset_info = dataset_info[dataset_id] 

        self.dataset = self._load_dataset()
        
        self.preprocess_fns = self.dataset_info['preprocess_fns']

        # Preprocess the dataset (generic):
        self._add_lemma_column()
        self._standardize_labels()

        # Preprocess the dataset (dataset specific):
        # ADD THEM TO DATASET_INFO AS preprocesss_fn = ['function1', 'function2', ...] and then call them here in a loop.
        self._apply_dataset_specific_preprocessing()

    def _load_dataset(self):
        # Load the dataset
        pass

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
    - lemma: the base lemma of the word (if the word is not a base lemma)
    - is_emoji: whether the word is an emoji or not
    """

    def __init__(self, lexicon_id, lexicon_path):
        self.lexicon_id = lexicon_id
        self.lexicon_path = lexicon_path
        self.lexicon_info = lexicon_info[lexicon_id]
        self.lexicon = self._load_lexicon()

        self.lemmatizer = WordNetLemmatizer()

        # Preprocess the lexicon:
        self._preprocess_lexicon()

    def print_info(self):
        print(f"Lexicon ID: {self.lexicon_id}")
        print(f"Lexicon Path: {self.lexicon_path}")
        print(f"Lexicon Info: {self.lexicon_info}")
        print(f"Lexicon Sample:\n{self.lexicon.head()}")

    def _load_lexicon(self):
        """Load lexicon from file based on configuration"""
        load_params = self.lexicon_info.get('load_params', {})
        return pd.read_csv(self.lexicon_path, **load_params)
    
    def _preprocess_lexicon(self):
        preprocess_fn = self.lexicon_info.get('preprocess_fn', None)
        if preprocess_fn:
            # Call the preprocessing function (defined in a separate file) on the lexicon dataframe.
            if preprocess_fn == 'hurtlex_preprocess':
                self._hurtlex_preprocess()

    def _hurtlex_preprocess(self):
        # The lexicon contains lemma but it is not lemmatized, so rename it as word
        self.lexicon.rename(columns={'lemma': 'word'}, inplace=True)
        # The lexicon contains word (POS) -> extract POS in a separate column
        self.lexicon['pos'] = self.lexicon['word'].apply(lambda x: x.split('(')[-1].split(')')[0] if '(' in x and ')' in x else None)
        self.lexicon['pos'] = self.lexicon['pos'].str.strip()
        # Standardize POS tags to 'n' for noun, 'v' for verb, 'a' for adjective 
        self.lexicon['pos'] = self.lexicon['pos'].replace({'adjective': 'a', 'noun': 'n', 'verb': 'v'})

        
        # Remove the POS from the word column
        self.lexicon['word'] = self.lexicon['word'].apply(lambda x: x.split('(')[0] if '(' in x else x)
        self.lexicon['word'] = self.lexicon['word'].str.strip()

        # Remove multiple word entries
        self._remove_multiple_word_entries()

        # Add lemma column
        self.lexicon['lemma'] = self.lexicon.apply(lambda row: self._return_base_lemma(row['word'], pos=row['pos']), axis=1)

        # Add is_emoji column
        self.lexicon['is_emoji'] = self.lexicon['word'].apply(self._is_emoji)
        
    def _is_emoji(self, token):
        return token in emoji.EMOJI_DATA
    
    def _remove_multiple_word_entries(self):
        self.lexicon['word'] = self.lexicon['word'].str.strip()
        multiple_word_entries = self.lexicon[self.lexicon['word'].str.contains(' ')]
        print(f"Removing {len(multiple_word_entries)} entries that are multiple words.")
        self.lexicon = self.lexicon[~self.lexicon['word'].str.contains(' ')]
    
    def _return_base_lemma(self, token, pos=None):
        if pos:
            lemma = self.lemmatizer.lemmatize(token, pos=pos)
            return lemma if lemma != token else token

        for fallback_pos in ('n', 'v', 'a'):
            lemma = self.lemmatizer.lemmatize(token, pos=fallback_pos)
            if lemma != token:
                return lemma

        return token
    
    


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
    
    

