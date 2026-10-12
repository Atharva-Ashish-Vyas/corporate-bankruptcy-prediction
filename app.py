"""Local demonstration app. Start with: python app.py"""
from pathlib import Path
import io
import json
import sys
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from bankruptcy.predict import load_bundle, predict_frame

def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
    bundle = load_bundle(ROOT)
    report = json.loads((ROOT / "reports/metrics.json").read_text())
    examples = json.loads((ROOT / "examples/demo_cases.json").read_text())

    @app.get("/")
    def index():
        return render_template("index.html", report=report, cases=examples)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model=bundle["name"])

    @app.get("/api/example/<int:case_id>")
    def example(case_id):
        if not 0 <= case_id < len(examples):
            return jsonify(error="Unknown demonstration case."), 404
        case = examples[case_id]
        result = predict_frame(bundle, pd.DataFrame([case["features"]])).to_dict(orient="records")[0]
        return jsonify(result=result, actual=case["actual"], label=case["label"], source_row=case["source_row"])

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("file")
        if upload is None or not upload.filename.lower().endswith(".csv"):
            return jsonify(error="Choose a CSV file with columns X1 through X64."), 400
        try:
            frame = pd.read_csv(io.BytesIO(upload.read()), na_values=["?"])
            if len(frame) > 5000:
                return jsonify(error="Use 5,000 rows or fewer for this demo."), 400
            result = predict_frame(bundle, frame)
            return jsonify(rows=result.to_dict(orient="records"), csv=result.to_csv(index=False))
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            return jsonify(error=str(exc)), 400

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="The file is too large. Use a CSV under 4 MB."), 413

    @app.get("/sample.csv")
    def sample():
        return send_from_directory(ROOT / "examples", "sample_companies.csv", as_attachment=True)

    @app.get("/figures/<path:filename>")
    def figures(filename):
        return send_from_directory(ROOT / "reports/figures", filename)

    return app

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
