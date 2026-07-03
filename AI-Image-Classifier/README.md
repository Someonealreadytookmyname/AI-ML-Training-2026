# AI Image Classifier

This is a Flask-based image classification app using the pre-trained VGG16 model from TensorFlow.
The app classifies uploaded images as Dog, Cat, or Other and displays the top-5 ImageNet predictions.

## Setup

1. Navigate to the project folder:
   ```bash
   cd 'GITHUB SHOWCASE/AI-Image-Classifier'
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Open your browser at:
   ```bash
   http://localhost:1323
   ```

## Notes

- The app loads the `VGG16` ImageNet model at startup.
- The frontend is served from `index.html`.
- The prediction endpoint is `POST /predict`.
- The index page uses `http://localhost:1323/predict` for uploads.
