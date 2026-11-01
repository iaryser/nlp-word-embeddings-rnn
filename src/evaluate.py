
import argparse
from pathlib import Path

import torch
import wandb

from models.embedding_classifier import EmbeddingClassifier
from models.lstm import LSTMClassifier
from models.gru import GRUClassifier
from data_loading import (
    load_embeddings,
    load_data,
    collate_fn_classifier,
    collate_fn_classifier_pair,
    collate_fn_lstm,
    collate_fn_lstm_pair,
)

PROJECT = "nlp-art-project-1"
EMBEDDING_DIM = 300
OUTPUT_DIM = 2


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--entity",
        type=str,
        required=True,
        help="Your W&B entity (username or team)"
    )

    parser.add_argument(
        "--run_ids",
        nargs=3,
        required=True,
        help="Training run IDs of the three selected models"
    )

    return parser.parse_args()


def build_model(config, embedding_matrix):
    """Reconstruct the model using its original training config."""

    if config["model"] == "classifier":
        input_mode = config.get("input_mode", "joint")
        if input_mode == "single":
            input_mode = "joint"
        model = EmbeddingClassifier(
            embedding_matrix,
            EMBEDDING_DIM * 4,
            config["classifier_hidden_dim"],
            OUTPUT_DIM,
            config["dropout"],
            input_mode=input_mode
        )

        collate_fn = collate_fn_classifier_pair if input_mode == "pair" else collate_fn_classifier

    elif config["model"] in ("lstm", "gru"):
        model_class = GRUClassifier if config["model"] == "gru" else LSTMClassifier
        model = model_class(
            embedding_matrix,
            EMBEDDING_DIM,
            config["hidden_dim"],
            config["classifier_hidden_dim"],
            OUTPUT_DIM,
            config["bidirectional"],
            dropout_rate=config["dropout"],
            input_mode=config["input_mode"]
        )

        if config["input_mode"] == "single":
            collate_fn = collate_fn_lstm
        elif config["input_mode"] == "pair":
            collate_fn = collate_fn_lstm_pair
        else:
            raise ValueError(
                f"Unknown input mode: {config['input_mode']}"
            )

    else:
        raise ValueError(
            f"Unknown model: {config['model']}"
        )

    return model, collate_fn


@torch.inference_mode()
def evaluate_model(model, test_loader, device):
    """Calculate classification accuracy on the test set."""

    model.eval()

    correct = 0
    total = 0

    for batch in test_loader:
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

        logits = model(batch)
        labels = batch["labels"]

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("The test dataset is empty.")

    return correct / total


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    api = wandb.Api()
    embedding_matrix = load_embeddings()

    results = []

    for index, run_id in enumerate(args.run_ids, start=1):

        # Retrieve the original training configuration
        training_run = api.run(
            f"{args.entity}/{PROJECT}/{run_id}"
        )

        config = dict(training_run.config)

        print(f"\nEvaluating model {index}/3")
        print(f"Training run: {training_run.name}")

        # Reconstruct model and select the correct collate function
        model, collate_fn = build_model(
            config,
            embedding_matrix
        )

        model = model.to(device)

        # Download the corresponding model artifact
        artifact_name = (
            f"{args.entity}/{PROJECT}/model-{run_id}:v0"
        )

        artifact = api.artifact(artifact_name)
        artifact_dir = Path(artifact.download())

        checkpoint_path = artifact_dir / f"{run_id}.pt"

        # Load trained model weights
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(state_dict)

        # Load test data using the original batch size
        test_loader = load_data(
            "Test",
            config["batch_size"],
            collate_fn
        )

        # Evaluate model
        test_accuracy = evaluate_model(
            model,
            test_loader,
            device
        )

        print(f"Test accuracy: {test_accuracy:.2%}")

        results.append({
            "run_id": run_id,
            "name": training_run.name,
            "accuracy": test_accuracy,
            "artifact": artifact_name,
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Log all evaluation results to a separate W&B run
    with wandb.init(
        entity=args.entity,
        project=PROJECT,
        job_type="evaluation",
        name="final-model-evaluation",
        config={"evaluated_run_ids": args.run_ids}
    ) as evaluation_run:

        for index, result in enumerate(results, start=1):
            evaluation_run.log({
                f"model_{index}/test_accuracy": result["accuracy"]
            })

            # Record the exact model artifact used
            evaluation_run.use_artifact(result["artifact"])

    # Print final comparison
    print("\nFinal test results")
    print("-" * 60)

    for result in results:
        print(
            f"{result['name']}: "
            f"{result['accuracy']:.2%}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
