import pandas as pd
from utils.data import StandardLexicon



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
    ########### Load and preprocess the lexicon ###########
    lexicon = StandardLexicon(lexicon_id='hurtlex_core_EN', lexicon_path=lexicon_path)
    lexicon.print_info()
   

    ########### Load and preprocess the datasets ###########




    

if __name__ == "__main__":
    main()