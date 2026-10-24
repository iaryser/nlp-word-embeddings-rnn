import nltk
import torch
import json
import gensim.downloader as api

from pathlib import Path
from datasets import load_dataset
from torch.nn import Embedding


wv = api.load('word2vec-google-news-300')

DATASET_KEYS = [
    "observation_1",
    "observation_2",
    "hypothesis_1",
    "hypothesis_2"
]

def tokenize_dataset(dataset: dict[str, list[str]]) -> list[dict[str, list[str]]]:
    processed_datset = []
    
    for i in range(len(dataset)):
        processed = {"label": dataset[i]["label"]}
        for key in DATASET_KEYS:
            processed[key] = nltk.word_tokenize(dataset[i][key])
        processed_datset.append(processed)
        
    return processed_datset

def normalize_token(word) -> str:
    if word in wv:
        return word
    if word.lower() in wv:
        return word.lower()
    if word.capitalize() in wv:
        return word.capitalize()
    return word

def merge_possessives(tokens) -> list[str]:
    merged = []

    for token in tokens:
        if token == "'s" and merged:
            merged[-1] += token
        else:
            merged.append(token)

    return merged

def process_tokens(dataset: list[dict[str, list]]) -> list[dict[str, list]]:
    processed_tokens = []
    
    for sample in dataset:
        processed_sample = sample.copy()
        for key in DATASET_KEYS:
            tokens = processed_sample[key]

            tokens = merge_possessives(tokens)

            tokens = [normalize_token(token) for token in tokens]

            processed_sample[key] = tokens
            
        processed_tokens.append(processed_sample)
    
    return processed_tokens

def build_vocabulary(dataset):
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for sample in dataset:
        for key in DATASET_KEYS:
            for token in sample[key]:
                if token not in vocab:
                    vocab[token] = len(vocab)

    return vocab

def build_embedding_matrix(vocab):
    embedding_dim = 300

    torch.manual_seed(42) #for reproducibility
    matrix = torch.randn(len(vocab), embedding_dim) * 0.01
    matrix[vocab["<PAD>"]] = 0

    for word, idx in vocab.items():
        if word in wv:
            matrix[idx] = torch.tensor(wv[word])

    return matrix

def encode_samples(vocab, samples):
    encoded = []

    for sample in samples:
        encoded_sample = sample.copy()

        for key in DATASET_KEYS:
            encoded_sample[key] = [
                vocab.get(token, vocab["<UNK>"])
                for token in sample[key]
            ]
        #take care of the labels to be 0 or 1 for their respective hypothesis
        encoded_sample["label"] = sample["label"] - 1 

        encoded.append(encoded_sample)

    return encoded
    
    
def encode_datasets(vocab, train_processed, valid_processed, test_processed):
    """
    Replace Tokens with vocab ID's in train/val/test
    Take care of Unknown words -> <UNK>
    """
    train_encoded = encode_samples(vocab, train_processed)
    valid_encoded = encode_samples(vocab, valid_processed)
    test_encoded = encode_samples(vocab, test_processed)
    
    return train_encoded, valid_encoded, test_encoded

def save_processed_data(
    vocab, embedding_matrix,
    train_encoded, valid_encoded, test_encoded
):
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save vocabulary
    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    # Save embedding matrix
    torch.save(embedding_matrix, output_dir / "embeddings.pt")

    # Save encoded datasets
    torch.save(train_encoded, output_dir / "train.pt")
    torch.save(valid_encoded, output_dir / "valid.pt")
    torch.save(test_encoded, output_dir / "test.pt")

def main():
    train = load_dataset("amarfurt/art-nlp-hs26", split="train")
    valid = load_dataset("amarfurt/art-nlp-hs26", split="validation")
    test = load_dataset("amarfurt/art-nlp-hs26", split="test")
    
    train_tokens = tokenize_dataset(train)
    valid_tokens = tokenize_dataset(valid)
    test_tokens = tokenize_dataset(test)
        
    processed_train_tokens = process_tokens(train_tokens)
    processed_valid_tokens = process_tokens(valid_tokens)
    processed_test_tokens = process_tokens(test_tokens)
    
    vocab = build_vocabulary(processed_train_tokens)
        
    embedding_matrix = build_embedding_matrix(vocab)
        
    enoced_train, encoded_valid, encoded_test = encode_datasets(
        vocab,
        processed_train_tokens, 
        processed_valid_tokens, 
        processed_test_tokens
    )
    
    save_processed_data(
        vocab,
        embedding_matrix,
        enoced_train,
        encoded_valid,
        encoded_test
    )

if __name__ == "__main__":
    main()