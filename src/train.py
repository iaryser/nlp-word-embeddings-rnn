import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.optim import Adam
import random
import argparse
import numpy as np

from models.embedding_classifier import EmbeddingClassifier
from models.lstm import LSTMClassifier
from data_loading import load_embeddings, load_data, collate_fn_classifier, collate_fn_lstm

embedding_dim = 300
output_dim = 2

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["classifier", "lstm"],
                        default="classifier")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--classifier_hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument(
        "--bidirectional",
        type=lambda x: x.lower() == "true",
        default=True
    )

    return parser.parse_args()

def main(args):
    seed = 42 #TODO: seed shuffling is also one thing i want to try out

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    run = wandb.init(
        project="nlp-art-project-1",
        config=vars(args)
    )

    config = wandb.config
    checkpoint_path = checkpoint_dir / f"{run.id}.pt"
    
    embedding_matrix = load_embeddings()
    
    if config.model == 'classifier':
        run.name = (
            f"{config.model}"
            f"-lr{config.learning_rate}"
            f"-drop{config.dropout}"
            f"-hidden{config.classifier_hidden_dim}"
        )
        
        model = EmbeddingClassifier(
            embedding_matrix,
            embedding_dim * 4,
            config.classifier_hidden_dim,
            output_dim,
            config.dropout
        )
        
        collate_fn = collate_fn_classifier

    else:
        run.name = (
            f"{config.model}"
            f"-lr{config.learning_rate}"
            f"-drop{config.dropout}"
            f"-hidden{config.hidden_dim}"
            f"-class_hidden{config.classifier_hidden_dim}"
            f"-bidi{config.bidirectional}"
        )
        
        model = LSTMClassifier(
            embedding_matrix,
            embedding_dim,
            config.hidden_dim,
            config.classifier_hidden_dim,
            output_dim,
            config.bidirectional,
            dropout_rate=config.dropout
        )
        
        collate_fn = collate_fn_lstm
        
    
    train_loader = load_data("Train", config.batch_size, collate_fn)
    valid_loader = load_data("Validation", config.batch_size, collate_fn)
    
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_valid_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(config.num_epochs):
        
        model.train()
        
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            logits = model(batch)
            labels = batch["labels"]

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total
        
        model.eval()

        valid_loss = 0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for batch in valid_loader:
                logits = model(batch)
                labels = batch["labels"]

                loss = criterion(logits, labels)

                valid_loss += loss.item() * labels.size(0)
                valid_correct += (logits.argmax(dim=1) == labels).sum().item()
                valid_total += labels.size(0)

        valid_loss /= valid_total
        valid_accuracy = valid_correct / valid_total
        
        # Log metrics to W&B
        wandb.log({
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "valid/loss": valid_loss,
            "valid/accuracy": valid_accuracy
        }, step=epoch + 1)


        # Save model if validation accuracy improves
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(model.state_dict(), checkpoint_path)

            print(f"New best model saved (valid accuracy: {valid_accuracy:.2%})")

        else:
            epochs_without_improvement += 1
        
        print( 
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Acc: {valid_accuracy:.2%}"
        )
        
        # Early stopping
        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
    wandb.summary["best_valid_accuracy"] = best_valid_accuracy
    wandb.summary["best_epoch"] = best_epoch
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    main(args)