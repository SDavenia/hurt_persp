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