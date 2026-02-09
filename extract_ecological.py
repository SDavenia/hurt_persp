import emoji
import nltk
import pandas as pd

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')

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

lexicon_path = 'data/lexicon/hurtlexcore_EN.tsv'
dataset_paths = {
    # dataset_id: dataset_path
}

def main():
    
    ########### Load and preprocess the lexicon (Move into StandardLexicon class) ###########
    lexicon = pd.read_csv('data/lexicon/hurtlexcore_EN.tsv', sep='\t', header=None, names=['id', 'lemma'])
    # The lexicon contains lemma (POS) -> extract POS in a separate column
    lexicon['pos'] = lexicon['lemma'].apply(lambda x: x.split('(')[-1].split(')')[0] if '(' in x and ')' in x else None)
    lexicon['pos'] = lexicon['pos'].str.strip()
    # Remove the POS from the lemma column
    lexicon['lemma'] = lexicon['lemma'].apply(lambda x: x.split('(')[0] if '(' in x else x)
    lexicon['lemma'] = lexicon['lemma'].str.strip()
    print(lexicon.head())

    # Remove all entries that are multiple words (i.e., contain spaces)
    print(f"Removing {len(lexicon[lexicon['lemma'].str.contains(' ')])} entries that are multiple words.")
    lexicon = lexicon[~lexicon['lemma'].str.contains(' ')]

    # Check that all entries are either emojis or valid lemmas.
    lemmatizer = WordNetLemmatizer()

    def is_base_lemma(token):
        return (
            lemmatizer.lemmatize(token, pos='n') == token and
            lemmatizer.lemmatize(token, pos='v') == token and
            lemmatizer.lemmatize(token, pos='a') == token
        )
    def is_emoji(token):
        return token in emoji.EMOJI_DATA
    
    def return_base_lemma(token):
        if is_base_lemma(token):
            return token
        else:
            return lemmatizer.lemmatize(token, pos='n') if lemmatizer.lemmatize(token, pos='n') != token else (lemmatizer.lemmatize(token, pos='v') if lemmatizer.lemmatize(token, pos='v') != token else lemmatizer.lemmatize(token, pos='a'))

    # Identify which entries are base lemmas and which are emojis, also return the base lemma for non-base lemma entries
    lexicon['is_base_lemma'] = lexicon['lemma'].apply(is_base_lemma)
    lexicon['base_lemma'] = lexicon['lemma'].apply(return_base_lemma)
    lexicon['is_emoji'] = lexicon['lemma'].apply(is_emoji)
    print("Found emojis number:", lexicon['is_emoji'].sum())
    
    # Print the entries that are not base lemmas or emojis
    print(lexicon[~(lexicon['is_base_lemma'] | lexicon['is_emoji'])])
    lexicon = lexicon[lexicon['is_base_lemma'] | lexicon['is_emoji']]

    # Load the dataset        




    

if __name__ == "__main__":
    main()