# Face Recognition — PCA + LDA + MLP

This repository originally contained only a notebook. It now includes a reproducible training and inference entrypoint. **No dataset or trained model is committed**, so predictions cannot run until a model has been trained using consented images.

## Local run

Use Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Add consented training photos in dataset/<person-name>/*.jpg
python train.py --dataset dataset --output artifacts/face_model.joblib
python predict.py path/to/test-photo.jpg --model artifacts/face_model.joblib
```

Training uses 100×100 grayscale images, PCA, LDA, then a two-hidden-layer MLP. A stratified test accuracy is printed when sufficient images exist; with too few images, no misleading accuracy is reported.

**Important:** This predicts among enrolled classes only. It cannot reliably identify unknown people, verify identities, or establish real-world biometric authentication. No face detection is included; images should be cropped to the face. Do not treat classifier probabilities as verified identity confidence.

## Deployment

Vercel can host a Next.js frontend, but not train this model on every request. For production inference, train offline, deploy the saved pipeline to a Python-serving backend with adequate resources, secure its endpoints, and connect your Vercel frontend via server-side environment variables. Do not publish the dataset or model artifacts containing sensitive biometric information. Set policies for consent, retention, and deletion.

The dataset and artifacts are intentionally excluded from version control.
