"""
Flask web application for real-time music genre prediction.
Upload an audio file and get genre classification results.
"""
import os
import tempfile

from flask import Flask, render_template, request, jsonify

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from webapp.inference import GenrePredictor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg"}

# Load predictor once at startup
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = GenrePredictor()
    return predictor


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Serve the upload form."""
    models = list(get_predictor().models.keys())
    return render_template("index.html", models=models)


@app.route("/predict", methods=["POST"])
def predict():
    """Handle audio upload and return genre prediction."""
    if "audio" not in request.files:
        return render_template("index.html", error="No file uploaded",
                               models=list(get_predictor().models.keys()))

    file = request.files["audio"]
    if file.filename == "":
        return render_template("index.html", error="No file selected",
                               models=list(get_predictor().models.keys()))

    if not allowed_file(file.filename):
        return render_template("index.html",
                               error=f"Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}",
                               models=list(get_predictor().models.keys()))

    model_name = request.form.get("model", "CNN")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = get_predictor().predict(tmp_path, model_name)
        return render_template("result.html", result=result)
    except Exception as e:
        return render_template("index.html", error=str(e),
                               models=list(get_predictor().models.keys()))
    finally:
        os.unlink(tmp_path)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic access."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    model_name = request.form.get("model", "CNN")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = get_predictor().predict(tmp_path, model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
