
import pandas as pd
from datasets import load_dataset

def load_and_prepare_data(dataset_name: str = "zeroshot/arxiv-biology"):
    print(f"Loading and preparing dataset: {dataset_name}...")
    
    dataset = load_dataset(dataset_name, split='train')
    df = dataset.to_pandas()
    df['full_text'] = df['title'] + " " + df['abstract']
    
    texts = df['full_text'].tolist()
    ids = df['id'].tolist()
    
    print(f"Total documents: {len(texts)}")
    return texts, ids

if __name__ == '__main__':
    texts, ids = load_and_prepare_data()
    print("\n Example:")
    print(f"ID: {ids[0]}")
    print(f"Text Snippet: {texts[0][:]}...")