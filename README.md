# Word Embeddings and RNNs

This project explores **Abductive Natural Language Inference (αNLI)** using pretrained Google News Word2Vec embeddings and recurrent neural networks.

The task is based on the *Abductive Reasoning in Narrative Text (ART)* dataset. Given two observations describing events at different points in time and two candidate hypotheses, the model must identify which hypothesis provides the more plausible explanation connecting the observations.

The project uses a [modified version of ART](https://huggingface.co/datasets/amarfurt/art-nlp-hs26), provided for the HSLU NLP course, with custom training, validation, and test splits.

Two architectures are compared:

- **Embedding Classifier:** Frozen Word2Vec embeddings with mean pooling, followed by a two-layer classifier with ReLU activation.
- **GRU Classifier:** Frozen Word2Vec embeddings followed by a two-layer unidirectional GRU and a two-layer classifier with ReLU activation.

## Results

The best configurations were selected based on validation accuracy and subsequently evaluated on the held-out test set.

| Model | Best Validation Accuracy | Test Accuracy |
|---|---:|---:|
| Embedding Classifier | 64.30% | 50.98% |
| GRU Classifier | **70.80%** | **54.90%** |

The GRU outperformed the embedding classifier on both validation and test data, although both models achieved lower accuracy on the test set.

### Best Hyperparameter Configurations

| Hyperparameter | Embedding Classifier | GRU Classifier |
|---|---:|---:|
| Learning rate | 0.001 | 0.001 |
| Batch size | 128 | 128 |
| Dropout | 0.5 | 0.3 |
| Classifier hidden dimension | 512 | 256 |
| GRU hidden dimension | — | 128 |
| Early stopping patience | 3 | 5 |
| Best epoch | 16 | 8 |

Both models were trained using the Adam optimizer and random seed 42, with a maximum of 30 epochs.

The complete configurations are available in [final_models.json](final_models.json). Training metrics, hyperparameter experiments, and model checkpoints are documented in the [W&B saved workspace](https://forge.coreweave.com/wandb/nlp-hs26-timo-ryser/nlp-art-project-1/workspace?nw=cehfc3y1vkuL).

## Setup

**Requirements:**
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Weights & Biases (W&B) account

Run all commands from the repository root. Internet access is required to download dependencies, the dataset, pretrained embeddings, and W&B artifacts. Several GB of RAM and disk space are recommended for the pretrained embeddings.

Install dependencies and configure W&B:

```sh
uv sync --locked
uv run python -m nltk.downloader punkt_tab
uv run wandb login
```

Copy `.env.example` to `.env` in the repository root and configure your W&B entity and project:

```dotenv
WANDB_ENTITY=YOUR_WANDB_ENTITY
WANDB_PROJECT=nlp-art-project-1
```

The commands below load these settings using `--env-file .env`.

The project uses PyTorch with CUDA 13.0 wheels. Training automatically uses a CUDA GPU when available and otherwise runs on the CPU.

## Reproduce the Results

Run the complete pipeline:

```sh
uv run --locked --env-file .env python pipeline.py --mode full
```

The pipeline performs the following steps:

1. **Preprocessing:** Downloads the `amarfurt/art-nlp-hs26` dataset and `word2vec-google-news-300` embeddings, then preprocesses the training, validation, and test splits.
2. **Training:** Trains both models using the configurations in `final_models.json` and saves the checkpoint with the highest validation accuracy for each model.
3. **Evaluation:** Evaluates both selected checkpoints on the test set and reports their accuracies.

Processed data is saved in `data/processed/` and local checkpoints in `checkpoints/`.

Training metrics, model artifacts, and evaluation results are logged to the configured W&B project. The pipeline also prints the corresponding training run IDs.

Exact results may vary slightly depending on hardware and the PyTorch environment.

## Rerun Individual Stages

### Training and Evaluation

If the preprocessed data already exists in `data/processed/`, skip preprocessing and run:

```sh
uv run --locked --env-file .env python pipeline.py --mode train
```

This trains and evaluates both models.

### Evaluation Only

To evaluate existing W&B checkpoints without retraining, provide their training run IDs:

```sh
uv run --locked --env-file .env python pipeline.py --mode evaluate --run-ids CLASSIFIER_RUN_ID GRU_RUN_ID
```

Ensure that the matching preprocessed data is available in `data/processed/` and that `.env` points to the W&B entity and project containing the checkpoints.

## References

- Bhagavatula, C., et al. (2020). *Abductive Commonsense Reasoning*. International Conference on Learning Representations (ICLR). [Paper](https://arxiv.org/abs/1908.05739)
- HSLU NLP HS26. [Modified ART dataset on Hugging Face](https://huggingface.co/datasets/amarfurt/art-nlp-hs26)