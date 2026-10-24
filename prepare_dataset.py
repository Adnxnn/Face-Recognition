"""Download the public Olivetti Faces research dataset and prepare training folders.

The dataset contains 40 subjects × 10 grayscale portraits. Downloading requires internet.
Use for research/demo only; do not infer consent for real-world biometric identification.
"""
from argparse import ArgumentParser
from pathlib import Path
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image
import numpy as np

def prepare(output: Path):
    dataset = fetch_olivetti_faces(shuffle=False, download_if_missing=True)
    for index, (face, label) in enumerate(zip(dataset.images, dataset.target)):
        target = output / f"subject_{int(label):02d}"
        target.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(np.uint8(np.clip(face, 0, 1) * 255), mode="L")
        image.save(target / f"image_{index:03d}.png")
    print(f"Prepared {len(dataset.images)} images of {len(np.unique(dataset.target))} people in {output}.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    prepare(args.output)
