## Code that loads different types of datasets and standardizes them into a common format.
## The common format should have:
## Text column: the text of the post/tweet/comment
## Label column: the label of the post/tweet/comment (need to select the specific dataset label)
## Sociodemographics column

import nltk
import emoji
import pandas as pd

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Dictionary containing text column name, label column name and sociodemographics column name for each dataset.
# Sociodemographics columns is either:
# - a single column 
# - a dict of columns with the trait of interest and the column name 
dataset_info = {
    'mhs': {
        'text_column': 'text',
        'label_column': 'hate_speech_score',
        
        # For now keep only a simplified one to see if code works
        'sociodemographics_columns': {
            'gender': 'annotator_gender',
            'education': 'annotator_educ',
            'income': 'annotator_income',
            'ideology': 'annotator_ideology',
            'age': 'annotator_age',
            'race': {
                'asian': 'annotator_race_asian',
                'black': 'annotator_race_black',
                'latinx': 'annotator_race_latinx',
                'middle_eastern': 'annotator_race_middle_eastern',
                'native_american': 'annotator_race_native_american',
                'pacific_islander': 'annotator_race_pacific_islander',
                'white': 'annotator_race_white'
            }
        },
        'preprocess_fns': 'mhs_preprocess',
        'load_params': {
            'sep': ',',
            'header': 0,
        }
    }

}

lexicon_info = {
    'hurtlex_core_EN': {
            'load_params': {
                'sep': '\t',
                'header': None,
                'names': ['id', 'lemma']
            },
            'preprocess_fn': 'hurtlex_preprocess'
        },
    'hurtlex_core_EN_clean': {
            'load_params': {
                'sep': ',',
                'header': None,
                'names': ['id', 'lemma']
            },
            'preprocess_fn': 'hurtlex_preprocess'
        }
    }


class StandardDataset:
    def __init__(self, 
                 dataset_id: str,
                 dataset_path: str):

        self.dataset_id = dataset_id
        self.dataset_path = dataset_path
        self.dataset_info = dataset_info[dataset_id] 

        self.dataset = self._load_dataset()
        self.preprocess_fns = self.dataset_info['preprocess_fns']
        # Rename the dataset-specific text column to text 
        self.dataset.rename(columns={self.dataset_info['text_column']: 'text'}, inplace=True)


        # TODO: Remove later: filter to keep only first 10000 rows to test code
        self.dataset = self.dataset.head(10000)
        
        # Strip the text from emojis and keep them in a separate column
        self.dataset['text_noemoji'], self.dataset['emojis'] = zip(*self.dataset['text'].apply(self.strip_emojis))        

        # Preprocess the dataset (same for all datasets, extract lemmas from the text column and add a lemma column containing a list of lemmas for each text):
        self._add_words_stems_lemmas_column()
        
        # Preprocess the dataset (dataset specific), need to preprocess sociodemographics and labels which are dataset specific:
        self._preprocess_dataset()

        # Keep only columns of interest
        self.dataset = self.dataset[['text', 'emojis', 'lemmas', 'lemmas_with_pos','words', 'stems', 'sociodemographics', 'label']]
        #self.dataset[['text', 'emojis', 'lemmas', 'words', 'stems', 'sociodemographics', 'label']].to_csv('try.csv', index=False)
        #raise ValueError("STOP HERE")
    
    def print_info(self):
        print(f"Dataset ID: {self.dataset_id}")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Dataset Info: {self.dataset_info}")
        print(f"Dataset Sample:\n{self.dataset.head(5)}")
        

    def _load_dataset(self):
        # Load the dataset
        return pd.read_csv(self.dataset_path, **self.dataset_info.get('load_params', {}))

    def strip_emojis(self, text):
        new_text = ""
        emojis = []
        for char in text:
            if char in emoji.EMOJI_DATA:
                emojis.append(char)
            else:
                new_text += char
        return new_text, emojis
    

    def _add_words_stems_lemmas_column(self):
        # lowercase -> tokenize -> compute pos -> lemmatize 
        # Adds a column with (lemma, pos) for each token in the text, needed for matching later.
        
        # TODO: Preprocessing the text? Like fuuuuuck -> fuck?
        # TODO: Add stopwords removal for faster matching downstream?
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()

        lemmas_list = [] 
        lemmas_with_pos_list = []
        words_list = []
        stems_list = []

        for i, text in enumerate(self.dataset['text_noemoji']):
            if i % 10000 == 0:
                print(f"Processing text {i}/{len(self.dataset)}")
            
            # Lowercase and tokenize + prepare pos tags
            tokens = word_tokenize(text.lower().strip())
            pos_tags_tb = pos_tag(tokens)

            # Remove all punctuation (not needed for us)
            tokens = [token for token in tokens if token.isalnum()]
            words_list.append(list(tokens))
            stems_list.append([stemmer.stem(token) for token in tokens])

            # Remove punctuation also here
            pos_tags_tb = [(token, pos) for token, pos in pos_tags_tb if token.isalnum()]
            all_forms = set()
            
            # Consider token, pos
            for token, pos in pos_tags_tb:
                # Convert pos to wordnet pos (n, v, a, r) for lemmatization
                pos_wn = self.treebank_to_wordnet_pos(pos)

                # If there was a pos tag, lemmatize with it
                if pos_wn is not None:
                    lemma = lemmatizer.lemmatize(token, pos=pos_wn)
                    pos_to_assign = pos_wn
                # Otherwise, rely on fallback with n->v->a->r
                else:
                    found_match = False
                    for fallback_pos in ('n', 'v', 'a', 'r'):
                        candidate = lemmatizer.lemmatize(token, pos=fallback_pos)
                        # Check if a lemma different from the word is found
                        if candidate != token:
                            lemma = candidate
                            pos_to_assign = fallback_pos
                            found_match = True
                            break
                        # Also check if the word is itself a valid lemma for the specified pos tag
                        elif wn.morphy(token, pos=fallback_pos) == token:
                            lemma = token
                            pos_to_assign = fallback_pos
                            found_match = True
                            break
                    # If no valid lemma is found
                    if not found_match:
                        lemma = token
                        pos_to_assign = None # TODO: Decide if we want to keep it as None or assign a default pos tag, like noun.

                all_forms.add((lemma, pos_to_assign))
            lemmas_list.append([lemma for lemma, pos in all_forms])
            lemmas_with_pos_list.append(list(all_forms))

        # Convert to sets for efficient matching later    
        self.dataset['lemmas'] = [set(x) for x in lemmas_list]
        self.dataset['lemmas_with_pos'] = [set(x) for x in lemmas_with_pos_list]
        self.dataset['words'] = [set(x) for x in words_list]
        self.dataset['stems'] = [set(x) for x in stems_list]

    def _preprocess_dataset(self):
        if self.preprocess_fns:
            if self.preprocess_fns == 'mhs_preprocess':
                self._mhs_preprocess()
    
    def _mhs_preprocess(self):
        ############## Label preprocessing ##############
        # Convert the label to binary: 
        # > 0.5 is approximately hate speech, < -1 is counter or supportive speech, and -1 to +0.5 is neutral or ambiguous.
        self.dataset['label'] = self.dataset[self.dataset_info['label_column']].apply(lambda x: 'hate_speech' if x > 0.5 else 'not_hate_speech')

        ############# Sociodemographics preprocessing ##############
        # For sociodemographics, for those with key: value -> extract the value, for those with key: {} -> extract from the boolean columns the traits that are True
        sociodemographics_columns = self.dataset_info['sociodemographics_columns']
        def _mhs_extract_sociodemographics(row):
            sociodemographics = {}
            for trait, columns in sociodemographics_columns.items():
                if isinstance(columns, str):
                    sociodemographics[trait] = row[columns]
                # TODO: Modify, if a person can assign multiple traits for the same sociodemographic category!
                elif isinstance(columns, dict):
                    all_valid_traits = []
                    for trait_value, column in columns.items():
                        if row[column] == 1:
                            all_valid_traits.append(trait_value)
                    if len(all_valid_traits) == 0:
                        sociodemographics[trait] = None
                    elif len(all_valid_traits) == 1:
                        sociodemographics[trait] = all_valid_traits[0]
                    else:
                        sociodemographics[trait] = all_valid_traits
                        # print(f"Warning: multiple traits found for trait {trait} in row {row.name}. Traits: {all_valid_traits}")
            return sociodemographics

        self.dataset['sociodemographics'] = self.dataset.apply(_mhs_extract_sociodemographics, axis=1)

    def treebank_to_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'  # Adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # Verb
        elif treebank_tag.startswith('N'):
            return 'n'  # Noun
        elif treebank_tag.startswith('R'):
            return 'r'  # Adverb
        else:
            return None


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
        # print(self.lexicon.head())

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

        # Preprocess the lexicon:
        self._preprocess_lexicon()

    def print_info(self):
        print(f"Lexicon ID: {self.lexicon_id}")
        print(f"Lexicon Path: {self.lexicon_path}")
        print(f"Lexicon Info: {self.lexicon_info}")
        print(f"Lexicon Sample:\n{self.lexicon.head(5)}")

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
        self.lexicon['pos'] = self.lexicon['pos'].replace({'adjective': 'a', 'noun': 'n', 'verb': 'v', 'adverb': 'r'})

        
        # Remove the POS from the word column
        self.lexicon['word'] = self.lexicon['word'].apply(lambda x: x.split('(')[0] if '(' in x else x)
        self.lexicon['word'] = self.lexicon['word'].str.strip()

        print(self.lexicon.head(10))

        # TODO: Remove multiple word entries, decide if we want to handle them later?
        self._remove_multiple_word_entries()

        # Add lemma column
        self.lexicon['lemma'] = self.lexicon.apply(lambda row: self._return_base_lemma(row['word'], pos=row['pos']), axis=1)

        # Add stem column
        self.lexicon['stem'] = self.lexicon['word'].apply(lambda x: self.stemmer.stem(x))

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
            return lemma 

        for fallback_pos in ('n', 'v', 'a'):
            lemma = self.lemmatizer.lemmatize(token, pos=fallback_pos)
            candidate = self.lemmatizer.lemmatize(token, pos=fallback_pos)
            if candidate != token:
                lemma = candidate
                return lemma
            elif wn.morphy(token, pos=fallback_pos) == token:
                lemma = token
                return lemma
        return token
    
    


class ExampleExtractor:
    """
    Takes as input StandardDataset, dataset id and StandardLexicon
    Return a pandas dataframe with columns:
     - text: the text of the post/tweet/comment
     - lemma: the lemma of the word in the lexicon that is present in the text
     - label: the label of the post/tweet/comment (need to select the specific dataset label)
     - sociodemographics: the sociodemographics of the post/tweet/comment (as a dict with {'trait1': value1, 'trait2': value2, ...})
     - dataset_source: the dataset id.
     - match_by: whether to extract and match by word, lemma or stem.
                 If word: extract the word from the text and match with the word column of the lexicon
                 If lemma: extract the lemma-pos pair from both the text and the lexicon and match them
                 If stem: extract the stem from the text and match with the stem column of the lexicon
    """

    def __init__(self,
                 standard_dataset: StandardDataset,
                 dataset_id: str,
                 standard_lexicon: StandardLexicon,
                 match_by: str = 'word'):
        
        self.dataset_id = dataset_id
        self.standard_dataset = standard_dataset
        self.standard_lexicon = standard_lexicon
        self.match_by = match_by

        self.results_df = self._extract_examples()
        print(f"Extracted {len(self.results_df)} examples from dataset {dataset_id} using match_by={match_by}.")



    def _extract_examples(self):
        non_emoji_lexicon = self.standard_lexicon.lexicon[~self.standard_lexicon.lexicon['is_emoji']]
        emoji_lexicon = self.standard_lexicon.lexicon[self.standard_lexicon.lexicon['is_emoji']]

        # Output 
        output_rows = []


        if self.match_by == 'word':
            lexicon_set = set(non_emoji_lexicon['word'])
        elif self.match_by == 'lemma':
            lexicon_set = set(non_emoji_lexicon['lemma'])
        elif self.match_by == 'lemma_with_pos':
            lexicon_set = set(non_emoji_lexicon.apply(lambda row: (row['lemma'], row['pos']), axis=1))
        elif self.match_by == 'stem':
            lexicon_set = set(non_emoji_lexicon['stem'])
        else:
            raise ValueError(f"Invalid match_by value: {self.match_by}. Must be one of 'word', 'lemma', 'lemma_with_pos' or 'stem'.")
        
        # Iterate through dataframe rows
        for idx, row in self.standard_dataset.dataset.iterrows():
            if idx % 1000 == 0:
                print(f"Processing row {idx}/{len(self.standard_dataset.dataset)}")

            # text matchings
            text = row['text']
            label = row['label']
            sociodemographics = row['sociodemographics']

            if self.match_by == 'word':
                matches = lexicon_set.intersection(row['words'])
                match_colname = 'word'
            elif self.match_by == 'lemma':
                matches = lexicon_set.intersection(row['lemmas'])
                match_colname = 'lemma'
            elif self.match_by == 'lemma_with_pos':
                matches = lexicon_set.intersection(row['lemmas_with_pos'])
                match_colname = 'lemma_with_pos'
            elif self.match_by == 'stem':
                matches = lexicon_set.intersection(row['stems'])
                match_colname = 'stem'
               
            for match in matches:
                output_rows.append({
                    'text': text,
                    match_colname: match,
                    'label': label,
                    'sociodemographics': sociodemographics,
                    'dataset_source': self.dataset_id
                })

            # emojis matchings
            df_emojis = row['emojis']
            for emoji in df_emojis:
                if emoji in set(emoji_lexicon['word']):
                    output_rows.append({
                        'text': text,
                        match_colname: emoji,
                        'label': label,
                        'sociodemographics': sociodemographics,
                        'dataset_source': self.dataset_id
                    })


        return pd.DataFrame(output_rows)

            

    
    

