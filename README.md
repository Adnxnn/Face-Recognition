# Face Recognition — PCA + LDA + MLP

A reproducible research/demo face-classification pipeline with dimensionality reduction and a multilayer perceptron. Originally a Jupyter notebook; now includes training, prediction, and dataset preparation scripts.

## Train using public research data

Python 3.11+ required. The Olivetti Faces dataset contains 400 face images from 40 subjects. The dataset is downloaded the first time the preparation script is run; internet is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_training.py
python predict.py dataset/subject_00/image_000.png
```

The trained model is written to `artifacts/face_model.joblib`. Training prints stratified holdout accuracy when sample counts permit. **Do not publish a biometric dataset or face model to a public repository.** The files are ignored by Git.

You can also use your own consented images under `dataset/<person-name>/*.jpg`, then run `python train.py --dataset dataset`. Do not mix Olivetti research identities with real enrolled users in a production system.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The included offline test uses *synthetic non-face inputs* only to check that the pipeline trains, saves and predicts. It does **not** establish recognition accuracy or model quality.

## Limitations and deployment

This classifies cropped faces among identities present in its training set; it is **not** real-time face detection, verified authentication, or reliable unknown-person rejection. Classifier probabilities are not calibrated identity assurances.

Train offline. To build an online demo, host secure inference in a persistent Python API with a trusted trained model; Vercel can host the user interface. Separate enrollment, data retention, consent, deletion policies and adequate identity verification safeguards are required for real-world applications.
