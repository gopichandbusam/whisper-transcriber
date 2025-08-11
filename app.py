from flask import Flask, render_template, request, send_file
import os
import whisper
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model once at startup.
# Using "base" model as it offers a good balance of speed and accuracy.
try:
    model = whisper.load_model("base")
except Exception as e:
    model = None
    print(f"Error loading Whisper model: {e}")

@app.route("/")
def index():
    """Renders the main page."""
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Handles file upload and transcription."""
    if model is None:
        return "Whisper model is not available. Please check the server logs.", 500

    if "file" not in request.files:
        return render_template("index.html", error="No file part in the request.")
    
    file = request.files["file"]
    
    if file.filename == "":
        return render_template("index.html", error="No file selected.")

    # Save the uploaded file
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Transcribe the audio
    try:
        result = model.transcribe(filepath, verbose=False)
    except Exception as e:
        return render_template("index.html", error=f"Error during transcription: {e}")

    # Generate timestamped text
    timestamped_text = ""
    for segment in result.get("segments", []):
        start = segment.get('start', 0.0)
        end = segment.get('end', 0.0)
        text = segment.get('text', '')
        timestamped_text += f"[{start:.2f}s -> {end:.2f}s] {text.strip()}\n"

    # Save the transcript to a file
    transcript_filename = f"{uuid.uuid4().hex}.txt"
    transcript_filepath = os.path.join(UPLOAD_FOLDER, transcript_filename)
    with open(transcript_filepath, "w", encoding="utf-8") as f:
        f.write(timestamped_text)

    return render_template(
        "index.html",
        transcription=timestamped_text,
        download_filename=transcript_filename
    )

@app.route("/download/<filename>")
def download(filename):
    """Provides the transcript file for download."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return "File not found.", 404
    return send_file(filepath, as_attachment=True, download_name="transcript.txt")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
