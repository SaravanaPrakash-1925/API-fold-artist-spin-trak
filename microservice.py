import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from spintrak_demo import (
    EMBED_DIM,
    MODEL_CONFIGS,
    DeterministicSPINTRAK,
    device,
    get_song_names,
    setup_deterministic_environment,
)
from utils import setup_logging

# FastAPI app
app = FastAPI(title="Audio Influence Score API")
logger = logging.getLogger(__name__)

# Global storage
MODELS: dict[str, nn.Module] = {}
EMBEDDINGS: dict[str, torch.Tensor] = {}
SONG_NAMES: dict[str, list[str]] = {}
GENRE_TO_MODEL = {
    key.split(":")[-1].replace("fold-", "").lower(): key for key in MODEL_CONFIGS
}


def init_single_model(model_name, config):
    logger.info(f"Loading resources for model: {model_name}")
    for resource, resource_name in [
        ("checkpoint", "file"),
        ("embedding", "file"),
        ("dataset", "path"),
    ]:
        path = config["checkpoint_path"]
        path = Path(path)
        logger.debug(f"Checking {resource}: {path}")
        resource = f"{resource.capitalize()} {resource_name}"
        if not path.exists():
            logger.error(f"{resource} not found: {path}")
            return
        logger.debug(f"{resource} found")

    logger.debug("Loading model...")
    model = MusicGen.get_pretrained(config["checkpoint_path"])

    MODELS[model_name] = model
    logger.info(f"Successfully loaded model for {model_name}")

    logger.debug("Loading embeddings...")
    embeddings = torch.load(config["embedding_path"], map_location=device).to(device)
    if embeddings.size(-1) != EMBED_DIM:
        logger.error(
            f"Loaded embeddings for {model_name} have dimension {embeddings.size()}, expected {EMBED_DIM}"
        )
        return
    EMBEDDINGS[model_name] = embeddings
    logger.info(f"Loaded embeddings for {model_name}, shape: {embeddings.shape}")

    logger.debug("Loading song names...")
    song_names = get_song_names(config["dataset_path"])
    if len(song_names) != embeddings.shape[0]:
        logger.warning(
            f"Mismatch for {model_name}: {len(song_names)} songs vs {embeddings.shape[0]} embeddings"
        )
        song_names = [f"Sample {i}" for i in range(embeddings.shape[0])]
    SONG_NAMES[model_name] = song_names
    logger.info(f"✓ Loaded {len(song_names)} song names for {model_name}")


def init_model_registry():
    for model_name, config in MODEL_CONFIGS.items():
        try:
            init_single_model(model_name, config)
        except Exception as e:
            logger.exception(f"✗ Error loading resources for {model_name}", exc_info=e)


# Response model
class SimilarTrack(BaseModel):
    rank: int
    similarity_score: float
    sample_index: int
    track: str
    percentage: float
    direction: str


# class InfluenceScoreResponse(BaseModel):
#     message: str
#     genre: str
#     top_similar_tracks: List[SimilarTrack]


class HealthCheckResponse(BaseModel):
    status: str
    loaded_models: List[str]
    available_models: List[str]
    model_details: Dict


@dataclass
class InfluenceScoreRequest:
    text_prompt: str = Form(...)
    audio_prompt: Optional[UploadFile] = File(None)
    genre: str = Form(...)
    # duration in seconds
    # TODO set an upper limit on the generation duration
    duration: int = Form(...)


# Custom middleware
@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    logger.info(f"Received request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(
            f"Completed request: {request.method} {request.url} - Status: {response.status_code}"
        )
    except Exception as e:
        logger.exception(
            f"Error in request: {request.method} {request.url}", exc_info=e
        )
        return JSONResponse(
            status_code=500, content={"detail": f"Middleware error: {e}"}
        )
    else:
        return response


@app.on_event("startup")
def startup_event():
    setup_deterministic_environment()
    print("=" * 60)
    print("🚀 STARTING UP AUDIO INFLUENCE SCORE API")
    print("=" * 60)

    logger.info("Starting up API, loading resources")

    # Log CUDA linear algebra backend info
    if torch.cuda.is_available():
        try:
            current_backend = torch.backends.cuda.preferred_linalg_library()
            logger.info(f"CUDA Linear Algebra Backend: {current_backend}")
        except Exception:
            logger.info("CUDA Linear Algebra Backend: Default")

    logger.info(f"Available MODEL_CONFIGS: {list(MODEL_CONFIGS.keys())}")
    logger.info(f"Using device: {device}")

    init_model_registry()

    logger.info(f"Startup complete. Successfully loaded models: {list(MODELS.keys())}")
    logger.info(f"Available embeddings: {list(EMBEDDINGS.keys())}")
    logger.info(f"Available song collections: {list(SONG_NAMES.keys())}")

    if not MODELS:
        logger.error("No models were loaded successfully!")
    else:
        logger.info(f"✓ Total models loaded: {len(MODELS)}")

    logger.info("API startup complete")


@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    logger.info("Health check requested")

    model_details = {}
    for model_name in MODEL_CONFIGS:
        model_details[model_name] = {
            "model_loaded": model_name in MODELS,
            "embeddings_loaded": model_name in EMBEDDINGS,
            "songs_loaded": model_name in SONG_NAMES,
            "embedding_shape": str(EMBEDDINGS[model_name].shape)
            if model_name in EMBEDDINGS
            else None,
            "num_songs": len(SONG_NAMES[model_name])
            if model_name in SONG_NAMES
            else None,
        }

    status = "healthy" if MODELS else "unhealthy"

    return HealthCheckResponse(
        status=status,
        loaded_models=list(MODELS.keys()),
        available_models=list(MODEL_CONFIGS.keys()),
        model_details=model_details,
    )


@app.post("/gradient")
async def compute_influence_scores_endpoint(request: InfluenceScoreRequest):
    logger.info(f"Received request for genre: {request.genre}")

    # Make genre case-insensitive by normalizing to lowercase
    genre = request.genre.lower()

    model_name = GENRE_TO_MODEL.get(genre)
    if model_name is None:
        logger.error(f"Invalid genre: {request.genre}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid genre '{request.genre}'. Choose from: {list(GENRE_TO_MODEL)} (case-insensitive)",
        )

    if model_name not in MODELS or model_name not in EMBEDDINGS:
        logger.error(f"Resources for {model_name} not loaded")
        raise HTTPException(
            status_code=500, detail=f"Resources for {model_name} not loaded"
        )

    # TODO better huge files handling
    audio_prompt = (
        None if request.audio_prompt is None else request.audio_prompt.file.read()
    )

    # TODO save audio prompts
    try:
        spintrak = DeterministicSPINTRAK(device=device)
        model = MODELS[model_name]
        results = spintrak.compute_influence_scores(
            model=model,
            audio_prompt=audio_prompt,
            text_prompt=request.text_prompt,
            train_embeddings=EMBEDDINGS[model_name],
            song_names=SONG_NAMES[model_name],
            top_k=10,
        )
        top_influences = results.get("top_influences", [])
        logger.info(
            f"Successfully processed request for {model_name} with top {len(top_influences)} positive influences"
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            saved = audio_write(
                path / "generated",
                results["generated_audio"][0].cpu(),
                model.sample_rate,
                strategy="loudness",
            )
            with saved.open("rb") as fh:
                audio = fh.read()
        # TODO add influence scores here
        return Response(content=audio, media_type="audio/wav")

    except Exception as e:
        logger.exception(f"Error processing request for {request.genre}", exc_info=e)
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {e}"
        ) from e

    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import uvicorn

    setup_logging()
    print("🌐 Starting FastAPI server on http://0.0.0.0:8002")
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8002)
