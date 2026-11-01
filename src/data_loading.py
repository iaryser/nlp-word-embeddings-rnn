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
SEP_TOKEN = 2
 
def load_embeddings():
    return torch.load("data/processed/embeddings.pt", weights_only=True)
    
def collate_fn_classifier(batch):
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


def collate_fn_rnn_pair(batch):
    sequences_1 = []
    sequences_2 = []
    
    for sample in batch:
        sequence_1 = (
            sample["observation_1"] + [SEP_TOKEN]
            + sample["hypothesis_1"] + [SEP_TOKEN]
            + sample["observation_2"]
        )
        
        sequence_2 = (
            sample["observation_1"] + [SEP_TOKEN]
            + sample["hypothesis_2"] + [SEP_TOKEN]
            + sample["observation_2"]
        )
        
        sequences_1.append(torch.tensor(sequence_1))
        sequences_2.append(torch.tensor(sequence_2))
        
    lengths_1 = torch.tensor([len(seq) for seq in sequences_1])
    lengths_2 = torch.tensor([len(seq) for seq in sequences_2])
    
    padded_1 = pad_sequence(
        sequences_1,
        batch_first=True,
        padding_value=PAD_TOKEN
    )
    
    padded_2 = pad_sequence(
        sequences_2,
        batch_first=True,
        padding_value=PAD_TOKEN
    )
    
    labels = torch.tensor(
        [sample["label"] for sample in batch],
        dtype=torch.long
    )
    
    return {
        "sequence_1": padded_1,
        "sequence_2": padded_2,
        "lengths_1": lengths_1,
        "lengths_2": lengths_2,
        "labels": labels
    }
        

def collate_fn_rnn(batch):
    sequences = []
    for sample in batch:
        sequence = (
            sample["observation_1"] + [SEP_TOKEN]
            + sample["hypothesis_1"] + [SEP_TOKEN]
            + sample["hypothesis_2"] + [SEP_TOKEN]
            + sample["observation_2"]
        )
        
        sequences.append(torch.tensor(sequence, dtype=torch.long))
    
    # Actual lengths of each sequence for the RNN
    lengths = torch.tensor([len(seq) for seq in sequences])    
    
    padded = pad_sequence(
        sequences,
        batch_first=True,# [batch_size, seq_len]
        padding_value=PAD_TOKEN
    )
    
    labels = torch.tensor(
        [sample["label"] for sample in batch],
        dtype=torch.long
    )
    
    return {
        "input_ids": padded,
        "lengths": lengths,
        "labels": labels
    }
        
def load_data(type: str, batch_size: int, collate_fn):
    
    datasets = {
    "Train": torch.load("data/processed/train.pt", weights_only=True),
    "Validation": torch.load("data/processed/valid.pt", weights_only=True),
    "Test": torch.load("data/processed/test.pt", weights_only=True)
    }

    return DataLoader(
        datasets[type],
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=(type == "Train")
    )

if __name__ == "__main__":
    train_loader = load_data("Train", 64, collate_fn_rnn)
    for batch in train_loader:
        break
