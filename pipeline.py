import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import secrets


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "final_models.json"


def run_command(command):
    print(f"\nRunning: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def preprocess():
    print("\n=== PREPROCESSING ===")
    run_command([sys.executable, "src/preprocessing.py"])


def load_model_configs():
    with open(CONFIG_FILE, encoding="utf-8") as file:
        configs = json.load(file)

    if not isinstance(configs, list) or not configs:
        raise ValueError("final_models.json must contain model configs.")

    return configs


def train_models():
    print("\n=== TRAINING FINAL MODELS ===")

    configs = load_model_configs()
    run_ids = []

    for index, config in enumerate(configs, start=1):
        run_id = secrets.token_hex(4)

        print(f"\nTraining model {index}/{len(configs)}")
        print(f"Run ID: {run_id}")

        command = [
            sys.executable,
            "src/train.py",
            "--run_id", run_id,
        ]

        for key, value in config.items():
            command.extend([
                f"--{key}",
                str(value).lower() if isinstance(value, bool)
                else str(value)
            ])

        run_command(command)
        run_ids.append(run_id)

    return run_ids


def evaluate_models(run_ids):
    print("\n=== EVALUATING MODELS ===")

    command = [
        sys.executable,
        "src/evaluate.py",
        "--run_ids", *run_ids,
    ]

    run_command(command)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["full", "train", "evaluate"],
        default="full"
    )

    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="Existing W&B run IDs for evaluation"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("WANDB_ENTITY") or not os.environ.get("WANDB_PROJECT"):
        raise ValueError(
            "Set WANDB_ENTITY and WANDB_PROJECT in .env and run with "
            "uv run --env-file .env."
        )

    if args.mode == "evaluate":
        expected_count = len(load_model_configs())
        if not args.run_ids or len(args.run_ids) != expected_count:
            raise ValueError(
                f"Evaluation requires exactly {expected_count} run IDs "
                "to match final_models.json."
            )

        evaluate_models(args.run_ids)
        return

    if args.mode == "full":
        preprocess()

    run_ids = train_models()

    evaluate_models(run_ids)

    print("\n=== PIPELINE COMPLETED ===")
    print("Training run IDs:", *run_ids)


if __name__ == "__main__":
    main()
