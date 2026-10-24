"""Run face inference with a previously trained model (not a live camera tracker)."""
from argparse import ArgumentParser
from pathlib import Path
import joblib
import numpy as np
from PIL import Image

def predict(model_path, image_path):
    payload = joblib.load(model_path)  # Only load artifacts you trained or explicitly trust.
    with Image.open(image_path) as img:
        sample = np.asarray(
            img.convert("L").resize(tuple(payload["image_size"])),
            dtype=np.float32
        ).reshape(1, -1) / 255.0
    model = payload["model"]
    predicted = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    # Confidence is a classifier score, not a calibrated identity verification result.
    return payload["names"][predicted], float(probabilities.max())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/face_model.joblib"))
    args = parser.parse_args()
    name, score = predict(args.model, args.image)
    print(f"Predicted identity: {name}; classifier score: {score:.3f}")
