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
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("podcast-backend")

# ====== Configuration ======
SPACE_ID = os.environ.get("HF_SPACE_ID", "NihalGazi/Text-To-Speech-Unlimited")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "podcasts")
TMP_ROOT = os.environ.get("TMP_ROOT", "/tmp/podcast_segments")
DEFAULT_STYLE = "Energetic sports commentary"
FALLBACK_VOICES = [
    "alloy","echo","fable","onyx","nova","shimmer","coral",
    "verse","ballad","ash","sage","amuch","dan"
]

# Ensure folders exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)

# Initialize the HF Client with retry logic
def get_client():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Initializing Gradio client (attempt {attempt+1}/{max_retries})")
            c = Client(SPACE_ID)
            logger.info("Client initialized successfully")
            return c
        except Exception as e:
            logger.error(f"Failed to initialize client: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

client = None

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
    # Fix common mojibake for apostrophes and dashes
    script = script.replace("â€™", "'").replace("â€\"", "-").replace("â€“", "-")
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
    global client
    
    # Ensure client is initialized
    if client is None:
        client = get_client()
    
    segment_paths = []
    for idx, (voice, text) in enumerate(assignments):
        logger.info("TTS call %d/%d – voice=%s text_preview=%s", idx+1, len(assignments), voice, text[:50])
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try calling with the exact parameters that worked in Colab
                result = client.predict(
                    prompt=text,
                    voice=voice,
                    emotion=style,
                    use_random_seed=True,
                    specific_seed=seed,
                    api_name="/text_to_speech_app"
                )
                
                # result should be a tuple/list with filepath at index 0
                if not result or len(result) == 0:
                    raise RuntimeError("TTS call returned empty result")
                
                remote_fp = result[0] if isinstance(result, (list, tuple)) else result
                
                if not remote_fp or not os.path.exists(remote_fp):
                    raise RuntimeError(f"TTS returned invalid filepath: {remote_fp}")
                
                # Move to our tmp_dir
                local_fp = os.path.join(tmp_dir, f"segment_{idx+1}.mp3")
                try:
                    shutil.move(remote_fp, local_fp)
                except Exception:
                    # Fallback: try copying if moving fails
                    shutil.copy(remote_fp, local_fp)
                
                segment_paths.append(local_fp)
                logger.info(f"Successfully generated segment {idx+1}")
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.error(f"TTS attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    # Try reinitializing client
                    try:
                        client = get_client()
                    except:
                        pass
                else:
                    raise RuntimeError(f"Failed to generate segment {idx+1} after {max_retries} attempts: {e}")
    
    return segment_paths


# ---------- API endpoints ----------
@app.on_event("startup")
async def startup_event():
    global client
    try:
        client = get_client()
    except Exception as e:
        logger.error(f"Failed to initialize client on startup: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "client_ready": client is not None}


@app.get("/voices")
def voices():
    """
    Return a list of available voices.
    """
    return {"voices": FALLBACK_VOICES}


@app.post("/generate-podcast")
def generate_podcast(payload: dict):
    """
    Blocking version: waits for all TTS + merge to finish, then returns URL
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

    logger.info(f"Processing {len(lines)} lines with voices: {voices}")
    assignments = alternate_assign_voices(lines, voices)

    job_id = str(uuid.uuid4())
    tmp_dir = os.path.join(TMP_ROOT, job_id)
    os.makedirs(tmp_dir, exist_ok=True)

    output_filename = f"{job_id}.mp3"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        # Generate all TTS segments (blocking)
        segment_paths = generate_segments_blocking(assignments, tmp_dir, style, seed)
        
        if not segment_paths:
            raise RuntimeError("No audio segments were generated")

        # Merge into single mp3
        merge_audio_files(segment_paths, output_path)
        
        if not os.path.exists(output_path):
            raise RuntimeError("Merged audio file was not created")

    except Exception as exc:
        logger.exception("Error generating podcast: %s", exc)
        safe_rm_tree(tmp_dir)
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(exc)}")
    finally:
        safe_rm_tree(tmp_dir)

    download_url = f"/download/{job_id}"
    return {"status": "completed", "audio_url": download_url, "job_id": job_id}


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
