# main.py
import os
import re
import uuid
import shutil
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
from gradio_client import Client
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("podcast-backend")

# ====== Configuration ======
SPACE_ID = os.environ.get("HF_SPACE_ID", "NihalGazi/Text-To-Speech-Unlimited")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "podcasts")
TMP_ROOT = os.environ.get("TMP_ROOT", "/tmp/podcast_segments")
DEFAULT_STYLE = "Energetic sports commentary"
# Available voices fallback list (frontend can use /voices)
FALLBACK_VOICES = [
    "alloy","echo","fable","onyx","nova","shimmer","coral",
    "verse","ballad","ash","sage","amuch","dan"
]

# Ensure folders exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)

# Initialize the HF Client once
client = Client(SPACE_ID)

app = FastAPI(title="Podcast TTS Backend")

# Allow CORS from your CREAO frontend origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount podcasts directory so files are directly accessible if desired
app.mount("/podcasts", StaticFiles(directory=OUTPUT_DIR), name="podcasts")

# Thread pool for blocking TTS calls
executor = ThreadPoolExecutor(max_workers=3)


# ---------- Utility functions ----------
def clean_script_simple(script: str) -> List[str]:
    """
    Simple script cleaning per S1:
    - remove timestamps [mm:ss] and [hh:mm:ss]
    - fix common encoding artifacts (basic)
    - split into lines, trim empties
    """
    if not script:
        return []
    # Remove timestamps like [00:07], [3:22], [01:22:33]
    script = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", script)
    # Fix common mojibake for apostrophes
    script = script.replace("â", "'").replace("â€”", "-")
    # Normalize whitespace and split lines
    lines = [ln.strip() for ln in script.splitlines()]
    # Remove empty lines and short noise lines
    lines = [ln for ln in lines if ln and len(ln) > 1]
    return lines


def alternate_assign_voices(lines: List[str], voices: List[str]) -> List[tuple]:
    """
    Return list of tuples (voice, text) for each line.
    If voices has one item, use it for all lines.
    If voices has >= 2, rotate across lines (voice0, voice1, voice0...).
    If voices has more than 2, cycle through them.
    """
    if not voices:
        voices = [ "nova" ]  # default fallback
    assigned = []
    if len(voices) == 1:
        for ln in lines:
            assigned.append((voices[0], ln))
    else:
        for i, ln in enumerate(lines):
            assigned.append((voices[i % len(voices)], ln))
    return assigned


def merge_audio_files(segment_paths: List[str], out_path: str,
                      pause_ms: int = 300):
    """
    Merge a list of audio files into a single mp3 file using pydub.
    """
    logger.info("Merging %d segments into %s", len(segment_paths), out_path)
    final = AudioSegment.silent(duration=200)
    for p in segment_paths:
        seg = AudioSegment.from_file(p)
        final += seg + AudioSegment.silent(duration=pause_ms)
    final.export(out_path, format="mp3")
    logger.info("Exported merged file: %s", out_path)


def safe_rm_tree(path: str):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception as e:
        logger.warning("Failed to remove tmp dir %s: %s", path, e)


# ---------- TTS worker (blocking) ----------
def generate_segments_blocking(assignments: List[tuple], tmp_dir: str,
                               style: str, seed: Optional[int] = 123) -> List[str]:
    """
    For each (voice, text) call the gradio client.predict(...) and save the resulting
    audio file inside tmp_dir. Returns list of local filepaths in order.
    """
    segment_paths = []
    for idx, (voice, text) in enumerate(assignments):
        logger.info("TTS call %d/%d – voice=%s text_preview=%s", idx+1, len(assignments), voice, text[:50])
        # Use the same api_name that worked in Colab
        result = client.predict(
            prompt=text,
            voice=voice,
            emotion=style,
            use_random_seed=True,
            specific_seed=seed,
            api_name="/text_to_speech_app"
        )
        # result[0] is the file path returned by gradio_client
        if not result or not result[0]:
            raise RuntimeError("TTS call failed or returned no filepath")
        remote_fp = result[0]
        # Move to our tmp_dir
        local_fp = os.path.join(tmp_dir, f"segment_{idx+1}.mp3")
        try:
            os.rename(remote_fp, local_fp)
        except Exception:
            # Fallback: try copying if renaming fails
            shutil.copy(remote_fp, local_fp)
        segment_paths.append(local_fp)
    return segment_paths


# ---------- API endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def voices():
    """
    Return a list of available voices. Try to query the space for options; fallback to hardcoded list.
    """
    try:
        api_view = client.view_api()
        # Try to parse available voices from view_api output (best-effort)
        # view_api() may return a string; attempt to find 'voice' choices
        # If parsing fails, return fallback list
        if isinstance(api_view, str):
            # crude parse for "Dropdown] voice: Literal['alloy', 'echo', ...]"
            m = re.search(r"voice: Literal\[(.*?)\]", api_view)
            if m:
                choices = [c.strip(" '\"") for c in m.group(1).split(",")]
                return {"voices": choices}
        # if it's already structured or unexpected, just return fallback
    except Exception as e:
        logger.debug("Could not parse client.view_api(): %s", e)
    return {"voices": FALLBACK_VOICES}


@app.post("/generate-podcast")
def generate_podcast(payload: dict, background_tasks: BackgroundTasks):
    """
    Main endpoint. Payload expected:
    {
      "script": "<raw script text>",
      "voices": ["nova", "onyx"],   # 1 or multiple voice names
      "style": "Energetic sports commentary",
      "seed": 123   # optional
    }
    """
    script = payload.get("script", "")
    voices = payload.get("voices", []) or []
    style = payload.get("style", DEFAULT_STYLE)
    seed = payload.get("seed", 123)

    if not script:
        raise HTTPException(status_code=400, detail="Missing 'script' in body.")

    # Clean & split lines (S1 simple cleaning)
    lines = clean_script_simple(script)
    if not lines:
        raise HTTPException(status_code=400, detail="No usable lines after cleaning.")

    assignments = alternate_assign_voices(lines, voices)

    job_id = str(uuid.uuid4())
    tmp_dir = os.path.join(TMP_ROOT, job_id)
    os.makedirs(tmp_dir, exist_ok=True)

    output_filename = f"{job_id}.mp3"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Run blocking TTS & merge in a threadpool to keep FastAPI responsive
    def worker():
        try:
            segment_paths = generate_segments_blocking(assignments, tmp_dir, style, seed)
            merge_audio_files(segment_paths, output_path)
        except Exception as exc:
            logger.exception("Error in TTS worker: %s", exc)
            # Clean up on failure
            safe_rm_tree(tmp_dir)
            # Remove partial output if exists
            if os.path.exists(output_path):
                os.remove(output_path)
            raise
        finally:
            # remove tmp segments directory
            safe_rm_tree(tmp_dir)

    # Add background task (executes after response is returned)
    # NOTE: Render's request timeout may kill long tasks; for production consider background worker.
    background_tasks.add_task(worker)

    # Provide immediate response with download URL path — frontend should poll /download or use /status
    download_url = f"/download/{job_id}"
    return JSONResponse({"job_id": job_id, "audio_url": download_url, "status": "processing" })


@app.get("/download/{job_id}")
def download(job_id: str):
    """
    If processing finished and file exists, return FileResponse. Else return status.
    """
    filepath = os.path.join(OUTPUT_DIR, f"{job_id}.mp3")
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg", filename=f"{job_id}.mp3")
    else:
        return JSONResponse({"status": "processing", "message": "Audio not ready yet. Poll again."}, status_code=202)
