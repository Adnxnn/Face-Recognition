"""Offline pipeline integration test with synthetic images.

This only checks training/prediction mechanics; it is NOT face recognition accuracy.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from train import train
from predict import predict

def test_training_and_inference(tmp_path: Path):
    rng = np.random.default_rng(42)
    for label, intensity in (("alpha", 45), ("beta", 210)):
        folder = tmp_path / "dataset" / label
        folder.mkdir(parents=True)
        for i in range(6):
            pixels = np.clip(rng.normal(intensity, 7, (100, 100)), 0, 255).astype("uint8")
            Image.fromarray(pixels).save(folder / f"{i}.png")
    model_path = tmp_path / "artifacts" / "model.joblib"
    train(tmp_path / "dataset", model_path)
    assert model_path.is_file()
    label, score = predict(model_path, tmp_path / "dataset" / "alpha" / "0.png")
    assert label in ("alpha", "beta")
    assert 0 <= score <= 1
