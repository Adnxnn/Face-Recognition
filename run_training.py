"""One-command public-data training: download Olivetti then train PCA → LDA → MLP."""
from argparse import ArgumentParser
from pathlib import Path
from prepare_dataset import prepare
from train import train

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/face_model.joblib"))
    args = parser.parse_args()
    prepare(args.dataset)
    train(args.dataset, args.model)
