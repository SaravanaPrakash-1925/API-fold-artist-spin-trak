import gc
import json
import logging
import os
import re
import time
import multiprocessing as mp
import requests
from pathlib import Path
import base64
from datetime import datetime

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from spin_trak import SPINTRAK, show_influence_percentages
from spintrak.gradients import worker  # Poland team's worker

# Setup logger
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI()

class GradientRequest(BaseModel):
    generated_song_url: str  # URL or base64 encoded audio data string
    prompt: str
    duration: float

# ---------------------------------------------------------------------
# Simple in‑memory counter for titles (per process)
# ---------------------------------------------------------------------
GENERATION_COUNTER = 0

def next_generation_title() -> str:
    global GENERATION_COUNTER
    GENERATION_COUNTER += 1
    return f"AI_Generation_{GENERATION_COUNTER:04d}"

# If you want to persist across restarts, use a file-based counter instead.

# ---------------------------------------------------------------------
# Helpers from your original script
# ---------------------------------------------------------------------
def get_song_names_from_training_directory(training_dir):
    logger.info(f"Getting song names from training directory: {training_dir}")
    directory = Path(training_dir)
    music_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac"}

    descriptions = {f.stem for f in directory.rglob("*.json")}
    music_files = []
    for file in directory.rglob("*"):
        if (file.is_file() and file.suffix.lower() in music_extensions and file.stem in descriptions):
            music_files.append(file)

    music_files_sorted = sorted(music_files, key=lambda f: f.name)
    song_names = [str(f) for f in music_files_sorted]
    logger.info(f"Found {len(song_names)} training songs with descriptions")
    return song_names

def load_training_gradients(bin_file_path):
    logger.info(f"Loading training gradients from: {bin_file_path}")
    data = np.fromfile(bin_file_path, dtype=np.float32)
    data = torch.from_numpy(data)
    embedding_dim = 8192
    num_samples = data.shape[0] // embedding_dim
    data = data.reshape(num_samples, embedding_dim)
    logger.info(f"Loaded training gradients shape: {data.shape} ({num_samples} samples, {embedding_dim} dims)")
    return data

def load_generated_gradients(pt_file_path):
    logger.info(f"Loading generated gradients from: {pt_file_path}")
    data = torch.load(pt_file_path, weights_only=False, map_location="cpu")
    if isinstance(data, list):
        if isinstance(data[0], dict) and 'gradients' in data[0]:
            gradients = data[0]['gradients']
        elif isinstance(data[0], torch.Tensor):
            gradients = data[0]
        else:
            raise ValueError(f"Unexpected list[0] type: {type(data[0])}")
    elif isinstance(data, dict):
        gradients = data.get('gradients', data.get('embeddings', data))
    elif isinstance(data, torch.Tensor):
        gradients = data
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

    if gradients.dim() == 1:
        gradients = gradients.unsqueeze(0)
    elif gradients.dim() == 3:
        gradients = gradients.squeeze(0) if gradients.shape[0] == 1 else gradients[0]
    logger.info(f"Extracted generated gradients shape: {gradients.shape}")
    return gradients

def display_all_influences(influence_scores, song_names):
    scores_flat = influence_scores.squeeze(0).cpu()
    all_percentages = show_influence_percentages(
        scores_flat,
        song_names,
        top_k=len(scores_flat)
    )
    return all_percentages

def generate_embedding_single(
    wav_path: Path,
    prompt: str,
    duration: float,
    output_dir: Path,
    model_path: str = None,
    gpu_id: int = 0,
):
    logger.info(f"Generating embedding for {wav_path.name} with prompt '{prompt[:40]}...' duration {duration}s")
    config = {
        "name": f"{wav_path.stem}_{'_'.join(prompt.strip().split()[:3])}_{int(duration)}",
        "seed": 0,
        "prompt": prompt,
        "duration": duration,
        "directory": output_dir,
    }

    task_queue = mp.Queue()
    task_queue.put(config)
    semaphore = mp.BoundedSemaphore(1)

    start_time = time.time()
    p = mp.Process(
        target=worker,
        args=(
            0,
            gpu_id,
            task_queue,
            semaphore,
            model_path,
            output_dir,
            duration,
            [],
            0,
        ),
    )
    p.start()
    p.join()
    elapsed = time.time() - start_time
    logger.info(f"Finished embedding generation process for {wav_path.name} in {elapsed:.2f}s")
    return elapsed  # gradient generation time

def clean_prompt_for_filename(prompt: str) -> str:
    prompt = prompt.lower()
    prompt = re.sub(r'[^a-z0-9]+', '_', prompt)
    return prompt[:30].strip('_')

# ---------------------------------------------------------------------
# MAIN ENDPOINT
# ---------------------------------------------------------------------
@app.post("/gradient")
async def compute_gradient(request: GradientRequest):
    temp_wav_path = None
    try:
        logger.info(
            f"Received gradient request: "
            f"[prompt='{request.prompt[:40]}...', url='{request.generated_song_url[:100]}...']"
        )

        output_dir = Path("/home/ubuntu/influence_gradients")
        output_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------------------
        # 1. Download or decode audio + track audio gen time (approx)
        # -----------------------------------------------------------------
        t_audio_start = time.time()
        if request.generated_song_url.startswith('http'):
            logger.info(f"Downloading audio from URL: {request.generated_song_url}")
            response = requests.get(request.generated_song_url, timeout=30)
            response.raise_for_status()
            audio_bytes = response.content
        else:
            logger.info("Decoding base64 audio data")
            audio_bytes = base64.b64decode(request.generated_song_url)

        temp_wav_path = output_dir / f"temp_{int(time.time())}.wav"
        with open(temp_wav_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Saved audio to temporary file: {temp_wav_path} ({len(audio_bytes)} bytes)")
        t_audio_end = time.time()
        audio_download_time = t_audio_end - t_audio_start

        # You know requested duration (seconds) from request.duration
        audio_length_seconds = float(request.duration)
        time_per_second_audio = (
            audio_download_time / audio_length_seconds if audio_length_seconds > 0 else 0.0
        )

        # -----------------------------------------------------------------
        # 2. Generate gradient embedding + time it
        # -----------------------------------------------------------------
        model_path = "/home/ubuntu/musicgen_finetunes/f8f265ca_7"
        checkpoint_name = Path(model_path).name

        t_grad_start = time.time()
        grad_gen_time = generate_embedding_single(
            temp_wav_path, request.prompt, request.duration, output_dir, model_path
        )
        t_grad_end = time.time()
        # grad_gen_time already measures process time; keep both if you want
        time_per_second_generated_gradients = (
            grad_gen_time / audio_length_seconds if audio_length_seconds > 0 else 0.0
        )

        # -----------------------------------------------------------------
        # 3. Find generated .pt file and load
        # -----------------------------------------------------------------
        cleaned_prompt = clean_prompt_for_filename(request.prompt)
        # Descriptive title: AI_Generation_XXXX
        title_str = next_generation_title()

        embedding_name = f"{temp_wav_path.stem}_{'_'.join(request.prompt.strip().split()[:3])}_{int(request.duration)}.pt"
        embedding_path = output_dir / embedding_name
        if not embedding_path.exists():
            logger.error(f"Embedding file not found after generation: {embedding_path}")
            raise HTTPException(status_code=500, detail="Failed to generate embedding .pt file")

        # -----------------------------------------------------------------
        # 4. Load training + generated gradients
        # -----------------------------------------------------------------
        training_bin_path = "/home/ubuntu/poland_teams_codebase/fold-artist-spin-trak/phase1_training_gradients_combined.bin"
        training_audio_dir = "/home/ubuntu/Phase-1"

        song_names = get_song_names_from_training_directory(training_audio_dir)
        training_gradients = load_training_gradients(training_bin_path)
        generated_gradients = load_generated_gradients(str(embedding_path))

        if len(song_names) != len(training_gradients):
            if len(song_names) > len(training_gradients):
                song_names = song_names[:len(training_gradients)]
            else:
                song_names.extend([f"Sample_{i}" for i in range(len(song_names), len(training_gradients))])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_gradients = training_gradients.to(device)
        generated_gradients = generated_gradients.to(device)

        # -----------------------------------------------------------------
        # 5. Influence scores
        # -----------------------------------------------------------------
        spintrak = SPINTRAK(device=device)
        logger.info("Starting influence score computation")
        t_inf_start = time.time()
        influence_scores = spintrak.get_attribution_scores(
            training_gradients, generated_gradients, lambda_=0.1
        )
        t_inf_end = time.time()
        logger.info(f"Influence scores computed: shape {influence_scores.shape}")
        influence_time = t_inf_end - t_inf_start

        all_influences = display_all_influences(influence_scores, song_names)

        scores_flat = influence_scores.squeeze(0).cpu()
        pos_indices = (scores_flat > 0).nonzero(as_tuple=True)[0]
        pos_scores = scores_flat[pos_indices]
        sum_pos = pos_scores.sum().item()
        normalized_percent = (
            (pos_scores * 100) / sum_pos if sum_pos != 0 else torch.zeros_like(pos_scores)
        )

        results = []
        for i, idx in enumerate(pos_indices):
            percent = normalized_percent[i].item()
            score = scores_flat[idx].item()
            name = song_names[idx] if idx < len(song_names) else f"Sample_{idx.item()}"
            results.append(
                {
                    "index": idx.item(),
                    "score": score,
                    "normalized_percentage": percent,
                    "name": os.path.basename(name),
                }
            )
        results.sort(key=lambda x: x["normalized_percentage"], reverse=True)
        top_20 = results[:20]

        # -----------------------------------------------------------------
        # 6. Metadata (with timing info and title)
        # -----------------------------------------------------------------
        # Placeholder: you don't have per-training-sample grad time; set a heuristic or 0
        time_per_second_training_gradients = 0.0

        gpu_model = (
            torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
        )

        metadata = {
            "model_used": "MusicGen",
            "finetune_checkpoint": model_path,
            "training_dataset_size": len(training_gradients),
            "gradient_dataset_size": len(training_gradients),
            "time_per_second_audio": round(time_per_second_audio, 4),
            "time_per_second_training_gradients": round(time_per_second_training_gradients, 4),
            "time_per_second_generated_gradients": round(time_per_second_generated_gradients, 4),
            "gpu_model": gpu_model,
            "prompt": request.prompt,
            "seed": 42,
            "title": title_str,  # AI_Generation_XXXX
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "audio_length_seconds": audio_length_seconds,
        }

        response = {
            "message": "Audio processed successfully",
            "genre": checkpoint_name,
            "metadata": metadata,
            "top_similar_tracks": [
                {
                    "rank": idx + 1,
                    "similarity_score": top_20[idx]['score'],
                    "sample_index": top_20[idx]['index'],
                    "track": top_20[idx]['name'],
                    "percentage": top_20[idx]['normalized_percentage'],
                    "direction": "positively",
                }
                for idx in range(len(top_20))
            ],
        }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response

    except Exception as e:
        logger.error(f"Error processing gradient request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_wav_path and temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)
            logger.info(f"Cleaned up temporary file: {temp_wav_path}")
