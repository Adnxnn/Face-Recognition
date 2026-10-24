"""Train and export the PCA → LDA → MLP face recognition pipeline.

Dataset layout:
dataset/
  person_a/photo1.jpg
  person_a/photo2.jpg
  person_b/photo1.jpg
  person_b/photo2.jpg
"""
from argparse import ArgumentParser
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

SIZE = (100, 100)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize(SIZE), dtype=np.float32).reshape(-1) / 255.0

def load_dataset(root):
    if not root.is_dir():
        raise ValueError(f"Dataset directory not found: {root}")
    labels, samples, names = [], [], []
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        photos = [p for p in sorted(person.iterdir()) if p.suffix.lower() in EXTENSIONS]
        valid = []
        for photo in photos:
            try:
                valid.append(pixels(photo))
            except (OSError, ValueError) as exc:
                print(f"Skipping {photo}: {exc}")
        if len(valid) < 2:
            print(f"Skipping {person.name}: at least 2 readable images required.")
            continue
        index = len(names)
        names.append(person.name)
        samples.extend(valid)
        labels.extend([index] * len(valid))
    if len(names) < 2:
        raise ValueError("At least two people with two valid images each are required.")
    return np.stack(samples), np.array(labels), names

def train(dataset, output):
    X, y, names = load_dataset(dataset)
    counts = np.bincount(y)
    # Use a holdout only when there are enough images from every person.
    use_holdout = counts.min() >= 3 and len(y) >= 2 * len(names) + 2
    if use_holdout:
        test_size = max(len(names), int(np.ceil(0.25 * len(y))))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test = y_test = None
        print("Too few examples for a stratified holdout; no evaluation accuracy reported.")

    n_pca = min(150, X_train.shape[0] - 1, X_train.shape[1])
    if n_pca < 2:
        raise ValueError("Not enough training samples for PCA.")
    model = Pipeline([
        ("pca", PCA(n_components=n_pca, whiten=True, svd_solver="randomized", random_state=42)),
        ("lda", LinearDiscriminantAnalysis()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=1)),
    ])
    model.fit(X_train, y_train)
    if X_test is not None:
        print(f"Holdout accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "names": names, "image_size": SIZE}, output)
    print(f"Saved trained pipeline to {output} ({len(names)} identities, {len(y)} images).")
    print("Only use this model with consented images. Treat outputs as uncertain predictions.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/face_model.joblib"))
    args = parser.parse_args()
    train(args.dataset, args.output)
