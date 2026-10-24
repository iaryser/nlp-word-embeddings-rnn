import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.optim import Adam
import random
import numpy as np

from models.embedding_classifier import EmbeddingClassifier
from data_loading import load_embeddings, load_data

batch_size = 64
embedding_dim = 300
inner_dim = 256
output_dim = 2
learning_rate = 0.001
num_epochs = 30
dropout_rate = 0.3

def main():
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = "word2vec-classifier-dropout-03"
    checkpoint_path = checkpoint_dir / f"{run_name}.pt"
    
    wandb.init(
    project="nlp-art-project-1",
    name=run_name,
    config={
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "inner_dim": inner_dim,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "dropout": dropout_rate,
        "optimizer": "Adam"
        }
    )
    
    model = EmbeddingClassifier(
        load_embeddings(),
        embedding_dim * 4,
        inner_dim,
        output_dim,
        dropout_rate
    )

    train_loader = load_data("Train", batch_size)
    valid_loader = load_data("Validation", batch_size)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_valid_accuracy = 0.0
    best_epoch = 0
    patience = 10
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        
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
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Acc: {valid_accuracy:.2%}"
        )
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        
    wandb.summary["best_valid_accuracy"] = best_valid_accuracy
    wandb.summary["best_epoch"] = best_epoch
    wandb.finish()

if __name__ == "__main__":
    main()