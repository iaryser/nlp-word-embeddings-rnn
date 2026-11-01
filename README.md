# Word Embeddings and RNNs

Abductive reasoning on the ART dataset using frozen Google News Word2Vec embeddings. The final models are an embedding-based classifier and a unidirectional GRU with paired inputs. Their hyperparameters and seeds are defined in [final_models.json](final_models.json).

## Setup

You need Python 3.12, `uv`, and a Weights & Biases (W&B) account. Run all commands from the repository root. Internet access is required for downloading dependencies, the dataset, embeddings, and W&B artifacts. Allow several GB of RAM and disk space for the pretrained embeddings.

```sh
uv sync --locked
uv run python -m nltk.downloader punkt_tab
uv run wandb login
```

Copy `.env.example` to `.env` in the repository root and set your W&B username or team and project name:

```dotenv
WANDB_ENTITY=YOUR_WANDB_ENTITY
WANDB_PROJECT=nlp-art-project-1
```

The commands below load these settings with `--env-file .env` for both training and evaluation. You can choose your own project name.

The project uses CUDA 13.0 PyTorch wheels. Training automatically uses a CUDA GPU when available and otherwise runs on the CPU.

## Reproduce the results

```sh
uv run --locked --env-file .env python pipeline.py --mode full
```

This command:

1. Downloads the `amarfurt/art-nlp-hs26` dataset and `word2vec-google-news-300` embeddings, then preprocesses the train, validation, and test splits.
2. Trains the two models in `final_models.json`, saving the checkpoint with the best validation accuracy for each model.
3. Evaluates both checkpoints on the test set and prints their test accuracies.

Processed data is saved in `data/processed/` and checkpoints in `checkpoints/`. Training metrics, model artifacts, and test results are logged to the W&B entity and project configured in `.env`. The pipeline also prints the training run IDs.

The configurations use seed 42. Exact results can vary across hardware and PyTorch environments.

## Rerun individual stages

If `data/processed/` already contains the preprocessed data, skip preprocessing and train and evaluate both models:

```sh
uv run --locked --env-file .env python pipeline.py --mode train
```

To evaluate existing W&B checkpoints without retraining, supply their two training run IDs:

```sh
uv run --locked --env-file .env python pipeline.py --mode evaluate --run-ids CLASSIFIER_RUN_ID GRU_RUN_ID
```

Keep the matching preprocessed data in `data/processed/` and the checkpoints' W&B entity and project in `.env` when evaluating existing checkpoints.
