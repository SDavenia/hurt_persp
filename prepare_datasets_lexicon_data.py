import dotenv
import os
import pandas as pd
from huggingface_hub import hf_hub_download
from datasets import load_dataset

def main():
    ##### HF TOKEN setup
    dotenv.load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    os.environ["HF_TOKEN"] = HF_TOKEN

    #################### Lexicons setup ####################
    # Add categories to Hurtlex core clean. 
    # Download the Hurtlex version with tags:
    # Check if hurtlexcore_EN_clean_with_tags.csv already exists, if not create it by merging the clean and the with_tags versions of Hurtlex.
    if not os.path.exists("data/lexicon/hurtlexcore_EN_clean_with_tags.csv"):
        url = "https://raw.githubusercontent.com/valeriobasile/hurtlex/refs/heads/master/lexica/EN/1.2/hurtlex_EN.tsv"
        hurtlex_with_tags = pd.read_csv(url, sep="\t", header=0)
        hurtlex_clean_notags = pd.read_csv("data/lexicon/hurtlexcore_EN_clean.csv", sep=",", header=None, names=["id", "word"])

        # Group categories as a set per id
        hurtlex_grouped = (
            hurtlex_with_tags
            .groupby('id')['category']
            .apply(lambda x: set(x))   # <-- set instead of list
            .reset_index()
        )

        # Merge with clean lexicon
        hurtlex_clean_with_tags = pd.merge(
            hurtlex_clean_notags,
            hurtlex_grouped,
            on='id',
            how='left'
        )
        unique_categories = set().union(*hurtlex_clean_with_tags['category'].dropna())

        print(
            f"Hurtlex clean with tags has a total of "
            f"{len(hurtlex_clean_with_tags)} entries, "
            f"with {len(unique_categories)} unique categories."
        )

        hurtlex_clean_with_tags['category'] = (
            hurtlex_clean_with_tags['category']
            .apply(lambda x: '|'.join(sorted(x)) if isinstance(x, set) else x)
        )

        hurtlex_clean_with_tags.to_csv("data/lexicon/hurtlexcore_EN_clean_with_tags.csv", index=False, sep="\t")
        
    # ##### Measuring Hate Speech dataset #####
    # mhs_folder = "data/datasets/mhs"
    # os.makedirs(mhs_folder, exist_ok=True)
    # ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    # ds = ds['train']
    # ds.to_csv(os.path.join(mhs_folder, "data.csv"), index=False)

    # print(f"Succesfully downloaded and saved Measuring Hate Speech dataset to {mhs_folder}")



    ##### Voiced dataset #####
    voiced_folder = "data/datasets/voiced"
    os.makedirs(voiced_folder, exist_ok=True)

    repo_id = "Lab-PL/voiced"
    files = ["train.parquet", "dev.parquet", "test.parquet"]

    # Download files from HF Hub
    local_paths = {}
    for fname in files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset"
        )
        local_paths[fname] = local_path

    # Load Parquet files into pandas DataFrames
    dfs = []
    for split in files:
        df = pd.read_parquet(local_paths[split])
        dfs.append(df)

    # Concatenate all splits
    full_df = pd.concat(dfs, ignore_index=True)

    # Save to CSV in your local folder
    csv_path = os.path.join(voiced_folder, "data.csv")
    full_df.to_csv(csv_path, index=False)

    print(f"Succesfully downloaded, concatenated and saved Voiced dataset to {csv_path}")

    ##### POPQUORN dataset #####
    # POPQUORN
    import requests

    url = "https://raw.githubusercontent.com/Jiaxin-Pei/Potato-Prolific-Dataset/main/dataset/offensiveness/raw_data.csv"
    output_file = "data/datasets/popquorn/data.csv"

    response = requests.get(url)
    response.raise_for_status()  # check for errors

    with open(output_file, "wb") as f:
        f.write(response.content)

    print(f"Downloaded POPQUORN data file as: {output_file}")


    ##### SBIC dataset #####
    sbic_folder = "data/datasets/sbic"
    # Check these files existst in the folder, if not raise an error
    assert os.path.exists(os.path.join(sbic_folder, "SBIC.v2.trn.csv")), f"File SBIC.v2.trn.csv not found in {sbic_folder}"
    assert os.path.exists(os.path.join(sbic_folder, "SBIC.v2.dev.csv")), f"File SBIC.v2.dev.csv not found in {sbic_folder}"
    assert os.path.exists(os.path.join(sbic_folder, "SBIC.v2.tst.csv")), f"File SBIC.v2.tst.csv not found in {sbic_folder}"
    df_trn = pd.read_csv('data/datasets/sbic/SBIC.v2.trn.csv')
    df_dev = pd.read_csv('data/datasets/sbic/SBIC.v2.dev.csv')
    df_tst = pd.read_csv('data/datasets/sbic/SBIC.v2.tst.csv')
    df = pd.concat([df_trn, df_dev, df_tst], ignore_index=True)
    df.to_csv(os.path.join(sbic_folder, "data.csv"), index=False)



if __name__ == "__main__":
    main()