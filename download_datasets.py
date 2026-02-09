from datasets import load_dataset

def main():
    ##### Measuring Hate Speech dataset #####
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    ds = ds['train']
    ds.to_csv('data/datasets/mhs/data.csv', index=False)


if __name__ == "__main__":
    main()