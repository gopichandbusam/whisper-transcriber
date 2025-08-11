from flask import Flask, render_template, request, send_file, jsonify, abort
import os
import uuid
import threading
import time
import whisper
import multiprocessing as mp
from typing import Dict, Any, List, Optional
import numpy as np  # type: ignore
import math
import json
try:
    import torch  # type: ignore
except Exception:  # noqa: BLE001
    torch = None  # Fallback if torch not available yet (will be installed per requirements)

"""
Whisper Transcriber - A web-based audio transcription tool using OpenAI Whisper

Author: Gopichand Busam
Website: https://gopichand.me
License: MIT

This Flask application provides a modern web interface for audio transcription
with real-time progress tracking, GPU acceleration support, and timestamped outputs.
"""

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "100"))  # 100 MB default
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm"}
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")

# Supported Whisper model sizes (update as library evolves)
MODEL_ORDER = ["tiny", "base", "small", "medium", "turbo", "large", "large-v2", "large-v3"]
SUPPORTED_MODELS = MODEL_ORDER.copy()
if MODEL_SIZE not in SUPPORTED_MODELS:
    SUPPORTED_MODELS.append(MODEL_SIZE)

# Lazy global base model (used only for baseline readiness; per-job workers load chosen model)
try:
    model = whisper.load_model(MODEL_SIZE)
except Exception:  # noqa: BLE001
    model = None  # Still allow app to start; workers will attempt load

# Job store (in-memory; for production use persistent / external store or task queue)
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
job_processes: Dict[str, mp.Process] = {}
logs_max = 500  # max log entries kept per job

# Default decoding/transcription options
DEFAULT_OPTIONS = {
    "task": "transcribe",   # or "translate"
    "temperature": 0.0,
    "beam_size": 5,
    "best_of": 5,
}
NUMERIC_BOUNDS = {
    "temperature": (0.0, 1.0),
    "beam_size": (1, 20),
    "best_of": (1, 20),
}

# Approximate CPU real-time factors (processing seconds per 1 second of audio)
# These are heuristic and will vary by hardware; used only for progress estimation.
MODEL_RTF_FACTORS_CPU = {
    "tiny": 0.4,      # faster than realtime
    "base": 0.6,
    "small": 0.9,
    "medium": 1.4,
    "turbo": 0.3,
    "large": 2.2,
    "large-v2": 2.0,
    "large-v3": 1.8,
}

# Rough GPU speed factors (smaller = faster) relative to audio duration
MODEL_RTF_FACTORS_GPU = {
    "tiny": 0.06,
    "base": 0.09,
    "small": 0.15,
    "medium": 0.32,
    "turbo": 0.07,
    "large": 0.50,
    "large-v2": 0.46,
    "large-v3": 0.42,
}

# Detect GPU and select active factors
IS_GPU = bool(torch and getattr(torch, 'cuda', None) and torch.cuda.is_available())
GPU_NAME = None
if IS_GPU:
    try:
        GPU_NAME = torch.cuda.get_device_name(0)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        GPU_NAME = "GPU"

MODEL_RTF_FACTORS = MODEL_RTF_FACTORS_GPU if IS_GPU else MODEL_RTF_FACTORS_CPU

# Approximate loaded memory footprint (GB) for display (very rough)
MODEL_MEMORY_GB = {
    "tiny": 1.0,
    "base": 1.3,
    "small": 2.5,
    "medium": 5.0,
    "turbo": 6.0,
    "large": 7.5,
    "large-v2": 7.5,
    "large-v3": 7.8,
}


def analyze_models(duration_seconds: float) -> List[Dict[str, Any]]:
    """Return ETA and rough RAM needs for each model given audio duration."""
    analysis: List[Dict[str, Any]] = []
    for m in SUPPORTED_MODELS:
        rtf = MODEL_RTF_FACTORS.get(m)
        eta = duration_seconds * rtf if rtf and duration_seconds is not None else None
        analysis.append(
            {
                "model": m,
                "eta_seconds": eta,
                "memory_gb": MODEL_MEMORY_GB.get(m),
            }
        )
    return analysis


def suggest_model(duration_seconds: Optional[float] = None) -> str:
    """Heuristic recommended model based on hardware and audio length."""
    if IS_GPU:
        total_mem = None
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        for m in reversed(SUPPORTED_MODELS):
            req = MODEL_MEMORY_GB.get(m)
            if total_mem and req and req * 1.1 <= total_mem:
                return m
        return MODEL_SIZE
    # CPU path
    if duration_seconds is None:
        return "base"
    for m in reversed(SUPPORTED_MODELS):
        rtf = MODEL_RTF_FACTORS_CPU.get(m)
        if rtf and duration_seconds * rtf <= 1800:  # aim for under ~30m processing
            return m
    return "tiny"

def _preprocess_audio(job_id: str):
    """Load audio into memory, save as .npy for faster model load, update progress incrementally."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return
    path = job.get("filepath")
    if not path or not os.path.exists(path):
        update_job(job_id, status="error", error="File missing during preprocessing")
        log_event(job_id, "Preprocessing failed: file missing")
        return
    try:
        update_job(job_id, prep_status="loading", prep_progress_pct=10.0)
        log_event(job_id, "Loading audio into memory")
        audio = whisper.load_audio(path)
        if audio.size == 0:
            update_job(job_id, status="error", error="Empty audio after load")
            log_event(job_id, "Preprocessing failed: empty audio")
            return
        duration_seconds = float(len(audio) / 16000.0)
        update_fields = {"prep_status": "normalizing", "prep_progress_pct": 60.0}
        with jobs_lock:
            job = jobs.get(job_id)
            if job and not job.get("duration_seconds"):
                update_fields["duration_seconds"] = duration_seconds
                expected_prep = min(20.0, max(2.0, duration_seconds * 0.06))
                update_fields.setdefault("prep_start_time", time.time())
                update_fields["prep_expected_total_seconds"] = expected_prep
        update_job(job_id, **update_fields)
        log_event(job_id, "Normalizing audio")
        # Whisper.load_audio already returns float32 mono 16000; ensure dtype
        audio = audio.astype('float32', copy=False)
        arr_path = path + ".npy"
        np.save(arr_path, audio)
        update_job(job_id, prep_status="cached", prep_progress_pct=95.0, cached_audio_path=arr_path)
        log_event(job_id, "Audio cached to numpy array")
        # Finalize
        update_job(job_id, status="ready", prep_progress_pct=100.0)
        log_event(job_id, "Preparation complete; ready to start")
    except Exception as e:  # noqa: BLE001
        update_job(job_id, status="error", error=f"Preprocess error: {e}")
        log_event(job_id, f"Preprocessing error: {e}")

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def secure_unique_path(original_name: str) -> str:
    base, ext = os.path.splitext(original_name)
    unique = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
    return os.path.join(UPLOAD_FOLDER, unique)

def _interprocess_write(job_id: str, payload: Dict[str, Any]):
    sidecar = os.path.join(UPLOAD_FOLDER, f".{job_id}.status")
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:  # noqa: BLE001
        pass

def _worker_progress(job_id: str, progress_pct: float, **fields):
    base = {"job_id": job_id, "status": "processing", "progress_pct": progress_pct, **fields}
    _interprocess_write(job_id, base)

def _worker_complete(job_id: str, status: str, **fields):
    base = {"job_id": job_id, "status": status, **fields}
    _interprocess_write(job_id, base)

def _worker_transcribe(job_id: str, filepath: str, original_filename: str, model_size: str, options: Dict[str, Any]):
    """Worker process with optional chunk-level progress for more accurate UI updates.

    If environment variable PROGRESS_MODE == 'chunks', audio longer than 35s is segmented into ~30s chunks
    and each decoded sequentially, emitting progress updates between chunks. This is less efficient than
    Whisper's internal batching but provides user-visible incremental progress without modifying library internals.
    """
    progress_mode = os.environ.get("PROGRESS_MODE", "estimate").lower()
    try:
        mdl = whisper.load_model(model_size)
        # Prefer cached numpy array if preprocessing created it
        cached_path = filepath + ".npy"
        if os.path.exists(cached_path):
            try:
                audio = np.load(cached_path)
            except Exception:
                audio = whisper.load_audio(filepath)
        else:
            audio = whisper.load_audio(filepath)
        if audio.size == 0:
            raise ValueError("Uploaded audio appears empty or unreadable.")
        transcribe_kwargs = {
            "task": options.get("task", DEFAULT_OPTIONS["task"]),
            "temperature": options.get("temperature", DEFAULT_OPTIONS["temperature"]),
            "beam_size": options.get("beam_size", DEFAULT_OPTIONS["beam_size"]),
            "best_of": options.get("best_of", DEFAULT_OPTIONS["best_of"]),
            "verbose": False,
        }
        language = options.get("language") or None
        if language:
            transcribe_kwargs["language"] = language

        duration_seconds = len(audio) / 16000.0
        accumulated_text: List[str] = []
        accumulated_segments: List[Dict[str, Any]] = []
        
        if progress_mode in {"chunks", "segments"} and duration_seconds > 5:
            if progress_mode == "chunks":
                window_sec = 30
            else:  # segments mode (finer granularity)
                window_sec = int(os.environ.get("SEGMENT_WINDOW_SEC", "5"))
                if window_sec < 2:
                    window_sec = 2
                if window_sec > 30:
                    window_sec = 30
            segment_len_samples = window_sec * 16000
            total_segments = math.ceil(len(audio) / segment_len_samples)
            for i in range(total_segments):
                start = i * segment_len_samples
                end = min((i + 1) * segment_len_samples, len(audio))
                segment_audio = audio[start:end]
                # Call model on raw samples for window
                result = mdl.transcribe(segment_audio, **transcribe_kwargs)
                seg_text = result.get("text", "").strip()
                if seg_text:
                    accumulated_text.append(seg_text)
                
                # Extract segments with timestamps adjusted for chunk offset
                chunk_start_time = start / 16000.0
                if result.get("segments"):
                    for seg in result["segments"]:
                        adjusted_seg = {
                            "start": seg["start"] + chunk_start_time,
                            "end": seg["end"] + chunk_start_time,
                            "text": seg["text"]
                        }
                        accumulated_segments.append(adjusted_seg)
                
                processed_until = end / 16000.0
                pct = min(100.0, (processed_until / duration_seconds) * 100.0)
                _worker_progress(job_id, pct, partial=True, mode=progress_mode)
            text_output = " ".join(accumulated_text).strip()
            all_segments = accumulated_segments
        else:
            result = mdl.transcribe(filepath, **transcribe_kwargs)
            text_output = result.get("text", "").strip()
            all_segments = result.get("segments", [])
        if not text_output:
            raise ValueError("Transcription returned empty text.")
        
        # Format output with timestamps
        def format_timestamp(seconds):
            """Convert seconds to MM:SS format"""
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"
        
        # Create timestamped transcript
        timestamped_lines = []
        if all_segments:
            for segment in all_segments:
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                if text:
                    timestamped_lines.append(f"[{start_time} - {end_time}] {text}")
        
        # Combine plain text and timestamped version
        if timestamped_lines:
            final_output = f"=== TRANSCRIPT ===\n{text_output}\n\n=== TIMESTAMPED TRANSCRIPT ===\n" + "\n".join(timestamped_lines)
        else:
            final_output = text_output
        
        filename_wo_ext = os.path.splitext(os.path.basename(original_filename))[0]
        txt_filename = f"{filename_wo_ext}.txt"
        txt_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex[:6]}_{txt_filename}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(final_output)
        _worker_complete(job_id, "done", transcript_path=txt_path, transcript_filename=txt_filename, progress_pct=100.0)
    except Exception as e:  # noqa: BLE001
        _worker_complete(job_id, "error", error=str(e))

def _interprocess_complete(job_id: str, status: str, **fields):  # Backwards compatibility alias
    _worker_complete(job_id, status, **fields)

def spawn_job_process(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job.get("status") not in {"ready", "queued"}:
            return False
        job["status"] = "processing"
        job["start_time"] = time.time()
        log_event(job_id, "Started processing")
    model_size = job.get("model_size", MODEL_SIZE)
    options = job.get("options", DEFAULT_OPTIONS)
    p = mp.Process(target=_worker_transcribe, args=(job_id, job["filepath"], job["original_filename"], model_size, options), daemon=True)
    job_processes[job_id] = p
    p.start()
    return True

def update_job(job_id: str, **fields):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(fields)

def log_event(job_id: str, message: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        logs: List[Dict[str, Any]] = job.setdefault("logs", [])  # type: ignore[assignment]
        logs.append({"ts": time.time(), "msg": message})
        if len(logs) > logs_max:
            # trim oldest
            del logs[: len(logs) - logs_max]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "": 
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400
    filepath = secure_unique_path(file.filename)
    file.save(filepath)
    size_bytes = os.path.getsize(filepath)
    duration_seconds = None
    try:
        # load_audio resamples to 16000 Hz
        audio = whisper.load_audio(filepath)
        duration_seconds = float(len(audio) / 16000.0)
    except Exception:  # noqa: BLE001
        pass
    analysis = analyze_models(duration_seconds or 0.0) if duration_seconds else []
    recommended = suggest_model(duration_seconds)
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "original_filename": file.filename,
            "filepath": filepath,
            "status": "preparing",
            "created_at": time.time(),
            "size_bytes": size_bytes,
            "logs": [],
            # Will be filled at start: model_size, options
            "duration_seconds": duration_seconds,
            "prep_status": "pending",
            "prep_progress_pct": 0.0,
            "suggested_model": recommended,
        }
        # Rough expected prep seconds heuristic (bounded)
        expected_prep = None
        if duration_seconds:
            expected_prep = min(20.0, max(2.0, duration_seconds * 0.06))  # 6% of audio length, clamped 2-20s
            jobs[job_id]["prep_expected_total_seconds"] = expected_prep
            jobs[job_id]["prep_start_time"] = time.time()
    log_event(job_id, f"Uploaded file ({size_bytes} bytes)")
    log_event(job_id, "Audio preprocessing queued")
    threading.Thread(target=_preprocess_audio, args=(job_id,), daemon=True).start()
    return jsonify({
        "job_id": job_id,
        "status": "preparing",
        "size_bytes": size_bytes,
        "duration_seconds": duration_seconds,
        "analysis": analysis,
        "recommended_model": recommended,
    }), 201

@app.route("/start/<job_id>", methods=["POST"])
def start(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job.get("status") not in {"ready", "queued"}:
        return jsonify({"error": f"Cannot start job in status {job.get('status')}"}), 409
    # Parse body JSON for model & options
    body = {}
    if request.data:
        try:
            body = request.get_json(force=True, silent=True) or {}
        except Exception:  # noqa: BLE001
            return jsonify({"error": "Invalid JSON body"}), 400
    chosen_model = body.get("model_size") or job.get("suggested_model") or MODEL_SIZE
    if chosen_model not in SUPPORTED_MODELS:
        return jsonify({"error": f"Unsupported model_size '{chosen_model}'"}), 400
    # Options
    raw_opts = body.get("options", {})
    options = DEFAULT_OPTIONS.copy()
    for k, v in raw_opts.items():
        if k in DEFAULT_OPTIONS:
            if isinstance(DEFAULT_OPTIONS[k], float):
                try:
                    v = float(v)
                except Exception:
                    return jsonify({"error": f"Option {k} must be float"}), 400
                lo, hi = NUMERIC_BOUNDS[k]
                if not (lo <= v <= hi):
                    return jsonify({"error": f"Option {k} out of range [{lo},{hi}]"}), 400
            elif isinstance(DEFAULT_OPTIONS[k], int):
                try:
                    v = int(v)
                except Exception:
                    return jsonify({"error": f"Option {k} must be int"}), 400
                lo, hi = NUMERIC_BOUNDS[k]
                if not (lo <= v <= hi):
                    return jsonify({"error": f"Option {k} out of range [{lo},{hi}]"}), 400
            options[k] = v
    language = body.get("language") or raw_opts.get("language") or body.get("options", {}).get("language")
    if language:
        options["language"] = language
    # Attach estimation metadata prior to queue
    duration_sec = job.get("duration_seconds")
    rtf = MODEL_RTF_FACTORS.get(chosen_model)
    expected_total_seconds = None
    if duration_sec and rtf:
        expected_total_seconds = duration_sec * rtf
    update_job(job_id, status="queued", model_size=chosen_model, options=options, expected_total_seconds=expected_total_seconds)
    log_event(job_id, f"Model selected: {chosen_model}")
    log_event(job_id, f"Options: { {k: options[k] for k in sorted(options)} }")
    log_event(job_id, "Job queued")
    # Spawn process (changes status to processing inside spawn)
    spawned = spawn_job_process(job_id)
    if not spawned:
        return jsonify({"error": "Failed to spawn process"}), 500
    return jsonify({"job_id": job_id, "status": "processing"}), 202

@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    status = job.get("status")
    if status in {"done", "error", "cancelled"}:
        return jsonify({"error": f"Job already {status}"}), 409
    proc = job_processes.get(job_id)
    if proc and proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
    update_job(job_id, status="cancelled")
    log_event(job_id, "Job cancelled")
    return jsonify({"job_id": job_id, "status": "cancelled"})

@app.route("/status/<job_id>")
def job_status(job_id: str):
    # Integrate sidecar completion check
    sidecar = os.path.join(UPLOAD_FOLDER, f".{job_id}.status")
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Update main job record once
            with jobs_lock:
                if job_id in jobs:
                    # Always merge sidecar (includes progress updates) unless already terminal
                    if jobs[job_id].get("status") not in {"done", "error", "cancelled"}:
                        update_job(job_id, **{k: v for k, v in data.items() if k != "job_id"})
                        if data.get("status") == "done":
                            log_event(job_id, "Transcription complete")
                        elif data.get("status") == "error":
                            log_event(job_id, f"Error: {data.get('error')}")
        except Exception:
            pass
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    # Derive progress if not explicitly set
    payload = {k: v for k, v in job.items() if k not in {"filepath", "logs"}}
    # Preparing phase: expose unified progress_pct + ETA
    if job.get("status") == "preparing":
        prep_pct = job.get("prep_progress_pct")
        if prep_pct is not None:
            payload["progress_pct"] = prep_pct
        expected_prep = job.get("prep_expected_total_seconds")
        start_prep = job.get("prep_start_time")
        if expected_prep and start_prep:
            elapsed_prep = time.time() - start_prep
            remaining_prep = max(0.0, expected_prep - elapsed_prep)
            payload["eta_seconds"] = remaining_prep
    if job.get("status") == "processing":
        expected = job.get("expected_total_seconds")
        start_time = job.get("start_time")
        now = time.time()
        if expected and start_time:
            elapsed = now - start_time
            if not payload.get("progress_pct"):
                # Fallback estimated progress
                payload["progress_pct"] = max(1.0, min(99.0, (elapsed / expected) * 100.0))
            remaining = max(0.0, expected - elapsed)
            payload["eta_seconds"] = remaining
        elif not payload.get("progress_pct"):
            payload["progress_pct"] = 0.0
    if job.get("status") == "done":
        payload["download_url"] = f"/download/{job_id}"
    return jsonify(payload)

@app.route("/logs/<job_id>")
def job_logs(job_id: str):
    since = int(request.args.get("since", 0))
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        entries: List[Dict[str, Any]] = job.get("logs", [])  # type: ignore[assignment]
        sliced = entries[since:]
        return jsonify({"next": since + len(sliced), "entries": sliced})

@app.route("/download/<job_id>")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404)
    if job.get("status") != "done":
        return jsonify({"error": "Job not completed"}), 409
    transcript_path = job.get("transcript_path")
    if not transcript_path or not os.path.exists(transcript_path):
        return jsonify({"error": "Transcript file missing"}), 500
    download_name = job.get("transcript_filename", os.path.basename(transcript_path))
    return send_file(transcript_path, as_attachment=True, download_name=download_name)

@app.route("/health")
def health():
    return {
        "status": "ok",
        "default_model": MODEL_SIZE,
        "supported_models": SUPPORTED_MODELS,
        "default_options": DEFAULT_OPTIONS,
        "numeric_bounds": NUMERIC_BOUNDS,
        "model_rtf_factors_active": MODEL_RTF_FACTORS,
        "model_rtf_factors_cpu": MODEL_RTF_FACTORS_CPU,
        "model_rtf_factors_gpu": MODEL_RTF_FACTORS_GPU,
        "model_memory_gb": MODEL_MEMORY_GB,
        "gpu": {"available": IS_GPU, "name": GPU_NAME},
        "suggested_model": suggest_model(),
    }

@app.route("/jobs")
def list_jobs():
    with jobs_lock:
        out = []
        for j in jobs.values():
            entry = {
                "id": j.get("id"),
                "original_filename": j.get("original_filename"),
                "status": j.get("status"),
                "model_size": j.get("model_size"),
                "created_at": j.get("created_at"),
                "duration_seconds": j.get("duration_seconds"),
                "progress_pct": j.get("progress_pct") or (j.get("prep_progress_pct") if j.get("status") == "preparing" else None),
            }
            if j.get("status") == "processing":
                expected = j.get("expected_total_seconds")
                start_time = j.get("start_time")
                if expected and start_time:
                    elapsed = time.time() - start_time
                    eta = max(0.0, expected - elapsed)
                    entry["eta_seconds"] = eta
            elif j.get("status") == "preparing":
                expected_prep = j.get("prep_expected_total_seconds")
                start_prep = j.get("prep_start_time")
                if expected_prep and start_prep:
                    elapsed_prep = time.time() - start_prep
                    entry["eta_seconds"] = max(0.0, expected_prep - elapsed_prep)
                else:
                    entry["eta_seconds"] = None
            elif j.get("status") == "queued":
                entry["eta_seconds"] = None
            out.append(entry)
    # sort newest first
    out.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return jsonify(out)

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)