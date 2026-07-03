import io
import time
import logging
import numpy as np

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image as keras_image

# ---------------- CONFIG ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)

log = logging.getLogger(__name__)

IMG_SIZE = (224, 224)

MAX_BYTES = 10 * 1024 * 1024

ALLOWED = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/jpg"
}

DOG_THRESHOLD = 0.10
CAT_THRESHOLD = 0.08

DOG_INDICES = list(range(151, 269))
CAT_INDICES = list(range(281, 286))

# ---------------- APP ---------------- #

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

# ---------------- MODEL ---------------- #

log.info("Loading VGG16 ImageNet weights ...")

model = VGG16(weights="imagenet", include_top=True)

log.info("VGG16 ready.")

# ---------------- FUNCTIONS ---------------- #

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    img = img.resize(IMG_SIZE, Image.LANCZOS)

    arr = keras_image.img_to_array(img)

    arr = np.expand_dims(arr, axis=0)

    arr = preprocess_input(arr)

    return arr


def classify(file_bytes):

    preds = model.predict(
        preprocess_image(file_bytes),
        verbose=0
    )[0]

    dog_prob = float(np.sum(preds[DOG_INDICES]))
    cat_prob = float(np.sum(preds[CAT_INDICES]))

    is_dog = dog_prob >= DOG_THRESHOLD
    is_cat = cat_prob >= CAT_THRESHOLD

    # ---------- LABEL ---------- #

    if is_dog and dog_prob >= cat_prob:
        label = "Dog"
        confidence = dog_prob

    elif is_cat and cat_prob > dog_prob:
        label = "Cat"
        confidence = cat_prob

    elif is_dog:
        label = "Dog"
        confidence = dog_prob

    elif is_cat:
        label = "Cat"
        confidence = cat_prob

    else:
        label = "Other"
        confidence = max(dog_prob, cat_prob)

    # ---------- CONFIDENCE ---------- #

    conf_pct = round(confidence * 100, 1)

    if label == "Other":
        conf_level = "N/A"

    elif conf_pct >= 80:
        conf_level = "High"

    elif conf_pct >= 55:
        conf_level = "Medium"

    else:
        conf_level = "Low"

    # ---------- SPLIT ---------- #

    total = dog_prob + cat_prob

    if total <= 0:
        total = 1e-9

    dog_pct = round((dog_prob / total) * 100, 1)
    cat_pct = round((cat_prob / total) * 100, 1)

    # ---------- TOP 5 ---------- #

    top5 = decode_predictions(
        np.expand_dims(preds, 0),
        top=5
    )[0]

    top5_out = []

    for i, (_, name, score) in enumerate(top5):

        top5_out.append({
            "rank": i + 1,
            "class": name.replace("_", " ").title(),
            "score": round(float(score) * 100, 2)
        })

    return {
        "label": label,
        "confidence": conf_pct,
        "conf_level": conf_level,
        "dog_pct": dog_pct,
        "cat_pct": cat_pct,
        "raw_dog": round(dog_prob, 5),
        "raw_cat": round(cat_prob, 5),
        "top5": top5_out
    }

# ---------------- ROUTES ---------------- #

@app.route("/")
def home():
    return send_file("index.html")


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": "VGG16-ImageNet",
        "ready": True
    })


@app.route("/predict", methods=["POST"])
def predict():

    t0 = time.time()

    if "file" not in request.files:
        return jsonify({
            "error": "No file field named 'file'."
        }), 400

    f = request.files["file"]

    if not f or not f.filename:
        return jsonify({
            "error": "Empty filename."
        }), 400

    mime = (f.content_type or "").lower().split(";")[0].strip()

    if mime not in ALLOWED:
        return jsonify({
            "error": f"Unsupported type '{mime}'."
        }), 415

    data = f.read()

    if not data:
        return jsonify({
            "error": "Empty file."
        }), 400

    try:

        result = classify(data)

        result["inference_ms"] = round(
            (time.time() - t0) * 1000
        )

        log.info(
            "predict %-6s conf=%.1f%% dog_raw=%.4f cat_raw=%.4f (%d ms)",
            result["label"],
            result["confidence"],
            result["raw_dog"],
            result["raw_cat"],
            result["inference_ms"]
        )

        return jsonify(result), 200

    except Exception as exc:

        log.exception("Prediction error")

        return jsonify({
            "error": str(exc)
        }), 500

# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    log.info("Server → http://localhost:1323")

    app.run(
        host="0.0.0.0",
        port=1323,
        debug=False
    )   