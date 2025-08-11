def _worker_transcribe_simple(job_id: str, filepath: str, original_filename: str, model_size: str, options: Dict[str, Any]):
    """Simplified worker to avoid multiprocessing hangs."""
    try:
        # Import whisper inside worker to avoid module loading issues
        import whisper
        import numpy as np
        import os
        import uuid
        
        # Load model
        mdl = whisper.load_model(model_size)
        
        # Load audio - prefer cached .npy
        cached_path = filepath + ".npy"
        if os.path.exists(cached_path):
            try:
                audio = np.load(cached_path)
            except Exception:
                audio = whisper.load_audio(filepath)
        else:
            audio = whisper.load_audio(filepath)
            
        if audio.size == 0:
            raise ValueError("Empty audio file")
            
        # Simple transcription without chunking
        result = mdl.transcribe(filepath, 
                               task=options.get("task", "transcribe"),
                               temperature=options.get("temperature", 0.0),
                               beam_size=options.get("beam_size", 1),
                               best_of=options.get("best_of", 1),
                               verbose=False)
        
        text_output = result.get("text", "").strip()
        if not text_output:
            raise ValueError("No transcription output")
            
        # Create output with timestamps
        segments = result.get("segments", [])
        output_lines = ["=== TRANSCRIPT ===", text_output, ""]
        
        if segments:
            output_lines.append("=== TIMESTAMPED TRANSCRIPT ===")
            for seg in segments:
                start_min = int(seg["start"] // 60)
                start_sec = int(seg["start"] % 60)
                end_min = int(seg["end"] // 60)
                end_sec = int(seg["end"] % 60)
                timestamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
                output_lines.append(f"{timestamp} {seg['text'].strip()}")
        
        # Save output
        filename_base = os.path.splitext(os.path.basename(original_filename))[0]
        txt_filename = f"{filename_base}.txt"
        upload_folder = os.path.dirname(filepath)
        txt_path = os.path.join(upload_folder, f"{uuid.uuid4().hex[:6]}_{txt_filename}")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
            
        # Write completion status
        _worker_complete(job_id, "done", 
                        transcript_path=txt_path, 
                        transcript_filename=txt_filename, 
                        progress_pct=100.0)
                        
    except Exception as e:
        _worker_complete(job_id, "error", error=str(e))
