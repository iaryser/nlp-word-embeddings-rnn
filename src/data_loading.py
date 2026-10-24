import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

DATASET_KEYS = [
    "observation_1",
    "observation_2",
    "hypothesis_1",
    "hypothesis_2"
]

PAD_TOKEN = 0
 
def load_embeddings():
    return torch.load("data/processed/embeddings.pt", weights_only=True)
    
def collate_fn(batch):
    padded = {}

    for key in DATASET_KEYS:
        sequences = [
            torch.tensor(sample[key], dtype=torch.long)
            for sample in batch
        ]

        padded[key] = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=PAD_TOKEN
        )

    padded["labels"] = torch.tensor(
        [sample["label"] for sample in batch],
        dtype=torch.long
    )

    return padded
        
def load_data(type: str, batch_size: int):
    
    datasets = {
    "Train": torch.load("data/processed/train.pt", weights_only=True),
    "Validation": torch.load("data/processed/valid.pt", weights_only=True),
    "Test": torch.load("data/processed/test.pt", weights_only=True)
    }
    
    loader = DataLoader(
        datasets[type],
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=(type == "Train")
    )
    
    return loader
    
if __name__ == "__main__":    
    train_loader = load_data("Train", 64)
    valid_loader = load_data("Validation", 64)
    test_loader = load_data("Test", 64)
    
    
    