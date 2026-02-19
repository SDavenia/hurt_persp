## Code that loads different types of datasets and standardizes them into a common format.
## The common format should have:
## Text column: the text of the post/tweet/comment
## Label column: the label of the post/tweet/comment (need to select the specific dataset label)
## Sociodemographics column

import ast
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
    },
    'annWithAttitudes': {
        'text_column': 'tweet',
        'label_column': 'off_avg',
        'text_id_column': 'postId',
        'annotator_id_column': 'annId',
        'sociodemographics_columns': {
            'gender': 'annotatorGender',
            'race': 'annotatorRace',
            'age': 'annotatorAge',
            'ideology': 'annotatorPolitics'
        },
        'preprocess_fns': 'annWithAttitudes_preprocess',
        'load_params': {
            'sep': ',',
            'header': 0,
        }
        
    },
    'voiced': {
        'text_column': 'comment_text',
        'label_column': 'PERSON_TOXIC_raw',
        'text_id_column': 'comment_id',
        'annotator_id_column': 'annotator_id',
        'sociodemographics_columns': {
            'gender': 'gender',
            'race': 'race',
            'age': 'age',
            'education': 'education',
            'ideology': 'annotator_political'
        },
        'preprocess_fns': 'voiced_preprocess',
        'load_params': {
            'sep': ',',
            'header': 0,
        }
    },
    'sbic': {
        'text_column': 'post',
        'label_column': 'offensiveYN',
        'text_id_column': 'HITId',
        'annotator_id_column': 'WorkerId',
        'sociodemographics_columns': {
            'gender': 'annotatorGender',
            'race': 'annotatorRace',
            'age': 'annotatorAge',
            'ideology': 'annotatorPolitics'
        },
        'preprocess_fns': 'sbic_preprocess',
        'load_params': {
            'sep': ',',
            'header': 0,
        }
    },
    'popquorn': {
        'text_column': 'text',
        'label_column': 'offensiveness',
        'text_id_column': 'instance_id',
        'annotator_id_column': 'user_id',
        'sociodemographics_columns': {
            'gender': 'gender',
            'race': 'race',
            'age': 'age',
            'education': 'education'
        },
        'preprocess_fns': 'popquorn_preprocess',
        'load_params': {
            'sep': ',',
            'header': 0,
        }
    }



}

lexicon_info = {
    'hurtlexcore_EN': {
            'load_params': {
                'sep': '\t',
                'header': None,
                'names': ['id', 'lemma']
            },
            'tags': None,
            'preprocess_fn': 'hurtlex_preprocess'
        },
    'hurtlexcore_EN_clean': {
            'load_params': {
                'sep': ',',
                'header': None,
                'names': ['id', 'lemma']
            },
            'tags': None,
            'preprocess_fn': 'hurtlex_preprocess'
        },
    'hurtlexcore_EN_clean_with_tags': {
            'load_params': {
                'sep': '\t',
                'header': 0,
            },
            'tags': 'category',
            'preprocess_fn': 'hurtlex_preprocess'
        }
    }



class StandardDataset:
    def __init__(self, 
                 dataset_id: str,
                 dataset_path: str,
                 preprocess: bool = True):

        self.dataset_id = dataset_id
        self.dataset_path = dataset_path
        self.dataset_info = dataset_info[dataset_id] 

        self.dataset = self._load_dataset()
        self.preprocess_fns = self.dataset_info['preprocess_fns']

        self._handle_duplicates()

        # Check duplicates in text_id and annotation id columns
        if self.dataset_info.get('text_id_column', None) and self.dataset_info.get('annotator_id_column', None):
            text_id_col = self.dataset_info['text_id_column']
            annotator_id_col = self.dataset_info['annotator_id_column']
            duplicates = self.dataset[self.dataset.duplicated(subset=[text_id_col, annotator_id_col], keep=False)]
            if len(duplicates) > 0:
                raise ValueError(f"Found {len(duplicates)} duplicate rows in the dataset based on {text_id_col} and {annotator_id_col}. Please investigate these duplicates further and remove them before proceeding.")

        #### VARIABLE NAMES STANDARDIZATION #### -> TODO: Move into a separate function for better readability
        # Rename the dataset-specific text column to text 
        self.dataset.rename(columns={self.dataset_info['text_column']: 'text'}, inplace=True)

        # Map annotator ids on 1 to num annotators
        annotator_ids_map = {annotator_id: f"{self.dataset_id}_ann-{idx}" for idx, annotator_id in enumerate(self.dataset[self.dataset_info['annotator_id_column']].unique())}
        self.dataset[self.dataset_info['annotator_id_column']] = self.dataset[self.dataset_info['annotator_id_column']].map(annotator_ids_map)
        # Rename column to 'annotator_id' for consistency
        self.dataset.rename(columns={self.dataset_info['annotator_id_column']: 'annotator_id'}, inplace=True)
        
        # Map text ids on 1 to num texts
        text_ids_map = {text_id: f"{self.dataset_id}_text-{idx}" for idx, text_id in enumerate(self.dataset[self.dataset_info['text_id_column']].unique())}
        self.dataset[self.dataset_info['text_id_column']] = self.dataset[self.dataset_info['text_id_column']].map(text_ids_map)
        # Rename column to 'text_id' for consistency
        self.dataset.rename(columns={self.dataset_info['text_id_column']: 'text_id'}, inplace=True)

        # Copy the original label column to a new column 'original_label' before preprocessing, to keep track of the original labels in case we want to do further analysis later.
        self.dataset['original_label'] = self.dataset[self.dataset_info['label_column']].copy()

        # If we only want to extract dataset statistics, no need to keep all those
        if preprocess:
            # TODO: Remove later: filter to keep only first 10 rows to test code
            # self.dataset = self.dataset.head(10)
            
            # Strip the text from emojis and keep them in a separate column
            self.dataset['text_noemoji'], self.dataset['emojis'] = zip(*self.dataset['text'].apply(self.strip_emojis))        

            # Preprocess the dataset (same for all datasets, extract lemmas from the text column and add a lemma column containing a list of lemmas for each text):
            self._add_words_stems_lemmas_column()
            
            # Preprocess the dataset (dataset specific), need to preprocess sociodemographics and labels which are dataset specific:
            self._preprocess_dataset()

            # Keep only columns of interest
            self.dataset = self.dataset[['annotator_id', 'text_id', 'text', 'emojis', 'lemmas', 'lemmas_with_pos','words', 'stems', 'sociodemographics', 'label', 'original_label']]
            #self.dataset[['text', 'emojis', 'lemmas', 'words', 'stems', 'sociodemographics', 'label']].to_csv('try.csv', index=False)
        else: 
            self._preprocess_dataset()
            self.dataset = self.dataset[["annotator_id", "text_id", "text", "original_label", "sociodemographics", "label"]]
    
    def print_info(self):
        print(f"Dataset ID: {self.dataset_id}")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Dataset Info: {self.dataset_info}")
        print(f"Dataset Sample:\n{self.dataset.head(5)}")
    
    def _handle_duplicates(self):
        label_col = self.dataset_info['label_column']
        text_col = self.dataset_info['text_column']
        annotator_id_col = self.dataset_info['annotator_id_column']
        text_id_col = self.dataset_info['text_id_column']

        # Find all rows which are duplicates based on label, text, annotator id and text_id
        duplicates = self.dataset[self.dataset.duplicated(subset=[label_col, text_col, annotator_id_col, text_id_col], keep=False)]
        # Keep only one row for each duplicate group, and drop the rest
        self.dataset = self.dataset.drop_duplicates(subset=[label_col, text_col, annotator_id_col, text_id_col], keep='first').reset_index(drop=True)

        # Find all rows which are duplicates based on annotator id and text id
        duplicates_all = self.dataset[self.dataset.duplicated(subset=[annotator_id_col, text_id_col], keep=False)]
        # Remove all of them, since we cannot be sure which one is the correct one
        self.dataset = self.dataset.drop_duplicates(subset=[annotator_id_col, text_id_col], keep=False).reset_index(drop=True)

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

            # TODO: Add stopwords removal for faster matching downstream? 
            
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
                        pos_to_assign = None # TODO?: Decide if we want to keep it as None or assign a default pos tag, like noun.

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
            elif self.preprocess_fns == 'annWithAttitudes_preprocess':
                self._annWithAttitudes_preprocess()
            elif self.preprocess_fns == 'voiced_preprocess':
                self._voiced_preprocess()
            elif self.preprocess_fns == 'sbic_preprocess':
                self._sbic_preprocess()
            elif self.preprocess_fns == 'popquorn_preprocess':
                self._popquorn_preprocess()
            else:
                raise ValueError(f"Invalid preprocess_fns value: {self.preprocess_fns}. Must be one of 'mhs_preprocess', 'annWithAttitudes_preprocess', 'voiced_preprocess', 'sbic_preprocess' or 'popquorn_preprocess'.")
    
    def _remove_none_in_label_column(self):
        # Remove rows with None in the label column, since we need the label for our analysis.
        initial_len = len(self.dataset)
        self.dataset = self.dataset[self.dataset[self.dataset_info['label_column']].notna()]
        print(f"Removed {initial_len - len(self.dataset)} rows with None in the label column.")


    def _extract_sociodemographics(self, row):
        sociodemographics = {}
        for trait, column in self.dataset_info['sociodemographics_columns'].items():
            if isinstance(column, str):
                sociodemographics[trait] = row[column]
            elif isinstance(column, dict):
                all_valid_traits = []
                for trait_value, col in column.items():
                    if row[col] == 1:
                        all_valid_traits.append(trait_value)
                if len(all_valid_traits) == 0:
                    sociodemographics[trait] = None
                elif len(all_valid_traits) == 1:
                    sociodemographics[trait] = all_valid_traits[0]
                else:
                    sociodemographics[trait] = set(all_valid_traits)
                    # print(f"Warning: multiple traits found for trait {trait} in row {row.name}. Traits: {all_valid_traits}")
        return sociodemographics

    def _annWithAttitudes_preprocess(self):
        ############## Label preprocessing ##############
        # Convert the label to binary: scores <2 treated as not offensive, scores >=2 treated as offensive 
        self._remove_none_in_label_column()
        self.dataset['label'] = self.dataset[self.dataset_info['label_column']].apply(lambda x: 'hurtful' if x >=2  else 'not_hurtful')

        ############# Sociodemographics preprocessing ##############
        # For sociodemographics, keep them as they are since they are already in a clean format.
        self.dataset['sociodemographics'] = self.dataset.apply(self._extract_sociodemographics, axis=1)
    
    def _mhs_preprocess(self):
        ############## Label preprocessing ##############
        # Convert the label to binary: 
        # > 0.5 is approximately hate speech, < -1 is counter or supportive speech, and -1 to +0.5 is neutral or ambiguous.
        self._remove_none_in_label_column()
        self.dataset['label'] = self.dataset[self.dataset_info['label_column']].apply(lambda x: 'hurtful' if x > 0.5 else 'not_hurtful')

        ############# Sociodemographics preprocessing ##############
        # For sociodemographics, for those with key: value -> extract the value, for those with key: {} -> extract from the boolean columns the traits that are True
        self.dataset['sociodemographics'] = self.dataset.apply(self._extract_sociodemographics, axis=1)

    def _voiced_preprocess(self):
        ############## Label preprocessing ##############
        # Convert the label to binary:
        # Label contains : "Not at all offensive", "Slightly offensive", "Moderately offensive", "Very offensive", "Extremely offensive".
        #                   Map "Not at all offensive" to not hurtful, and "Slightly offensive", "Moderately offensive", "Very offensive", "Extremely offensive" to hurtful.
        # Check that all labels are in the expected set
        self._remove_none_in_label_column()
        expected_labels = {"Not at all offensive", "Slightly offensive", "Moderately offensive", "Very offensive", "Extremely offensive"}
        unique_labels = set(self.dataset[self.dataset_info['label_column']].unique())
        if not unique_labels.issubset(expected_labels):
            raise ValueError(f"Unexpected labels found in the dataset: {unique_labels - expected_labels}. Expected labels are: {expected_labels}")
        self.dataset['label'] = self.dataset[self.dataset_info['label_column']].apply(lambda x: 'hurtful' if x in ["Slightly offensive", "Moderately offensive", "Very offensive", "Extremely offensive"] else 'not_hurtful')

        ############# Sociodemographics preprocessing ##############
        # For sociodemographics, keep them as they are since they are already in a clean format.
        self.dataset['sociodemographics'] = self.dataset.apply(self._extract_sociodemographics, axis=1)

    def _sbic_preprocess(self):
        ############## Label preprocessing ##############
        # Convert the label to binary:
        # Label contains 1 for offensive, 0 for not offensive and 0.5 for maybe, keep 0.5 and 1 as hurtful and 0 as not hurtful.
        # Check that all labels are in the expected set
        self._remove_none_in_label_column()
        expected_labels = {0, 0.5, 1}
        unique_labels = set(self.dataset[self.dataset_info['label_column']].unique())
        if not unique_labels.issubset(expected_labels):
            raise ValueError(f"Unexpected labels found in the dataset: {unique_labels - expected_labels}. Expected labels are: {expected_labels}")
        self.dataset['label'] = self.dataset[self.dataset_info['label_column']].apply(lambda x: 'hurtful' if x in [0.5, 1] else 'not_hurtful')

        ############# Sociodemographics preprocessing ##############
        # For sociodemographics, keep them as they are since they are already in a clean format.
        self.dataset['sociodemographics'] = self.dataset.apply(self._extract_sociodemographics, axis=1)

    def _popquorn_preprocess(self):
        ############## Label preprocessing ##############
        # Convert the label to binary:
        # Dataset contains a score from 1 to 5 indicating the offensiveness of the comment, keep comments with score >= 2 as hurtful and < 2 as not hurtful.
        self._remove_none_in_label_column()
        self.dataset['label'] = self.dataset[self.dataset_info['label_column']].apply(lambda x: 'hurtful' if x >= 2 else 'not_hurtful')

        ############# Sociodemographics preprocessing ##############
        # For sociodemographics, keep them as they are since they are already in a clean format.
        self.dataset['sociodemographics'] = self.dataset.apply(self._extract_sociodemographics, axis=1)
       

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

        # TODO?: Remove multiple word entries, decide if we want to handle them later?
        self._remove_multiple_word_entries()

        # Add lemma column
        self.lexicon['lemma'] = self.lexicon.apply(lambda row: self._return_base_lemma(row['word'], pos=row['pos']), axis=1)

        # Add lemma_with_pos column
        self.lexicon['lemma_with_pos'] = self.lexicon.apply(lambda row: (row['lemma'], row['pos']), axis=1)

        # Add stem column
        self.lexicon['stem'] = self.lexicon['word'].apply(lambda x: self.stemmer.stem(x))

        # Add is_emoji column
        self.lexicon['is_emoji'] = self.lexicon['word'].apply(self._is_emoji)

        # If tag is not None, include a tag column with the tags as a set, they are saved as a string with | as separator in the original csv, so we need to split them and convert to set for easier matching later.
        if self.lexicon_info.get('tags', None):
            # Rename the tag column to 'tag' for consistency
            self.lexicon.rename(columns={self.lexicon_info['tags']: 'tag'}, inplace=True)
            # Convert the tag column to a set of tags
            self.lexicon['tag'] = self.lexicon['tag'].apply(lambda x: set(x.split('|')) if pd.notna(x) else set())
            
        
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
    
    def _remove_duplicates(self, match_by):
        # There can be multiple rows with same lemma_with_pos but different tags, so we merge them by grouping by lemma_with_pos and merging the tags into a set.
        if 'tag' in self.lexicon.columns:
            if match_by == 'lemma_with_pos':
                self.lexicon = self.lexicon.groupby('lemma_with_pos').agg({
                    'id': 'first',
                    'word': 'first',
                    'pos': 'first',
                    'lemma': 'first',
                    'stem': 'first',
                    'is_emoji': 'first',
                    'tag': lambda x: set.union(*x)  # Merge tags into a set
                }).reset_index()
            else:
                raise NotImplementedError("STOP")



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
        
        # There could be duplicates after lemmatization, e.g. hurtlex contains both dog and dogs
        self.standard_lexicon._remove_duplicates(match_by=match_by)

        self.lexicon_has_tags = self.standard_lexicon.lexicon_info.get('tags', None) is not None

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
            text_id = row['text_id']
            ann_id = row['annotator_id']
            original_label = row['original_label']
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
                    'dataset_source': self.dataset_id,
                    'annotator_id': ann_id,
                    'text_id': text_id,
                    'original_label': original_label
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
                        'dataset_source': self.dataset_id,
                        'annotator_id': ann_id,
                        'text_id': text_id,
                        'original_label': original_label
                    })
        # If the lexicon has tags, add the tags to the output dataframe by merging with the lexicon dataframe on the matched word/lemma/word_with_lemma/stem column.
        if self.lexicon_has_tags:
            output_df = pd.DataFrame(output_rows)
            output_df = output_df.merge(self.standard_lexicon.lexicon[[match_colname, 'tag']], how='left', left_on=match_colname, right_on=match_colname)
        else:
            output_df = pd.DataFrame(output_rows)
            
        return output_df

########### POSTPROCESSING FUNCTIONS: TODO: Move to the dataset class preprocess function for better readability ###########
def sanitize_none_string(dict_string):
    # Prefer not to say should be mapped to None
    if "'Prefer not to say'" in dict_string:
        dict_string = dict_string.replace("'Prefer not to say'", "None")
    
    if 'nan' in dict_string:
        dict_string = dict_string.replace('nan', 'None')
    if "'na'" in dict_string:
        dict_string = dict_string.replace("'na'", "None")
    
    dictionary = ast.literal_eval(dict_string)
    if dictionary.get('ideology', None):
        dictionary['ideology'] = str(dictionary['ideology']) if dictionary['ideology'] is not None else None
    
    return dictionary

def map_sociodemographics(socio_dict, sociodemographic_mappings):
    if not isinstance(socio_dict, dict):
        return None
    
    mapped_dict = {}
    
    for key, value in socio_dict.items():
        if key not in sociodemographic_mappings:
            mapped_dict[key] = None
            continue
        
        mapping = sociodemographic_mappings[key]
        
        # Validate non-age fields
        if key != 'age':
            if value is not None and str(value) not in mapping:
                raise ValueError(f"Unmapped value for {key}: {value}")
            mapped_value = mapping.get(str(value), None)
        
        # Special handling for age
        else:
            new_value = value
            if value is None:
                mapped_value = None

            else:
                # If it is string, first convert to numeric and then apply age mappings.
                if type(value) == str:
                    new_value = mapping.get(value, None)

                age = float(new_value)
                if age >= 100:
                    mapped_value = None
                elif age < 25:
                    mapped_value = "18-24"
                elif age < 35:
                    mapped_value = "25-34"
                elif age < 45:
                    mapped_value = "35-44"
                elif age < 55:
                    mapped_value = "45-54"
                elif age < 65:
                    mapped_value = "55-64"
                else:
                    mapped_value = "65 or older"
            
        mapped_dict[key] = mapped_value
        
    return mapped_dict