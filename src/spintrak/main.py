import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import typer

from fold_audiocraft.data.audio import audio_write
from fold_audiocraft.models import MusicGen

from spintrak.generate_audio import generate_with_musicgen
from spintrak.gradients import (
    create_training_dataset,
    create_training_dataset_parallel,
    get_gradient_embeddings,
)
from spintrak.initialization import load_model, setup_deterministic_environment
from spintrak.utils import LOCAL_MODEL_DIR, setup_logging

fold_audiocraft = importlib.import_module("fold_audiocraft")
sys.modules["audiocraft"] = fold_audiocraft
setup_logging()

app = typer.Typer()


@app.command()
def generate_audio(
    prompt: str = typer.Option(
        None, "--prompt", help="Text prompt to guide generation"
    ),
    duration: float = typer.Option(
        None, "--duration", help="Output duration in seconds"
    ),
    model_path: str = typer.Option(
        None, "--model", help="Path to directory containing specific MusicGen model"
    ),
):
    setup_deterministic_environment()
    if model_path:
        model = MusicGen.get_pretrained(model_path)
    else:
        model = load_model(LOCAL_MODEL_DIR, prompt)
    wav = generate_with_musicgen(model, prompt=prompt, output_duration=duration)
    for idx, one_wav in enumerate(wav):
        audio_write(f"{idx}", one_wav.cpu(), model.sample_rate, strategy="loudness")


@app.command()
def generate_gradients(
    prompt: str = typer.Option(
        None, "--prompt", help="Text prompt to guide generation"
    ),
    duration: float = typer.Option(
        None, "--duration", help="Output duration in seconds"
    ),
    output_path: str = typer.Option(
        ..., "--output", help="Path to projected gradients file"
    ),
    model_path: str = typer.Option(
        None, "--model", help="Path to directory containing specific MusicGen model"
    ),
):
    start = time.perf_counter()
    setup_deterministic_environment()
    if model_path:
        model = MusicGen.get_pretrained(model_path)
    else:
        model = load_model(LOCAL_MODEL_DIR, prompt)
    wav, gradients = get_gradient_embeddings(
        model, prompt=prompt, output_duration=duration, gpu_id=0
    )
    logging.debug("Saving gradients to file")
    torch.save(gradients, output_path)
    for idx, one_wav in enumerate(wav):
        audio_write(
            f"{Path(output_path).stem}",
            one_wav.cpu(),
            model.sample_rate,
            strategy="loudness",
        )
    end = time.perf_counter()
    logging.info(f"Total elapsed time: {end - start:.2f}s")


@app.command()
def generate_training_embedings(
    input_dir: str = typer.Option(
        ..., "--input", help="Path to directory contains music files and descriptions"
    ),
    output_path: str = typer.Option(
        ..., "--output", help="Path to training dataset gradients file"
    ),
    resume: int = typer.Option(
        False,
        "--resume",
        help="Flag that, when set, prevents overwriting existing files.",
    ),
    model_path: str = typer.Option(
        None, "--model", help="Path to directory containing specific MusicGen model"
    ),
):
    setup_deterministic_environment()
    if model_path:
        model = MusicGen.get_pretrained(Path(model_path))
    else:
        model = MusicGen.get_pretrained(Path(LOCAL_MODEL_DIR) / "fb-large/")
    create_training_dataset(model, input_dir, output_path, resume)


@app.command()
def generate_training_dataset_parallel(
    input_dir: str = typer.Option(
        ..., "--input", help="Path to directory contains music files and descriptions"
    ),
    output_path: str = typer.Option(
        ..., "--output", help="Path to training dataset gradients file"
    ),
    resume: int = typer.Option(
        False,
        "--resume",
        help="Flag that, when set, prevents overwriting existing files.",
    ),
    model_path: Optional[str] = typer.Option(None, "--model", help="..."),
    duration: float = typer.Option(
        None, "--duration", help="Output duration in seconds"
    ),
    processes_per_gpu: int = typer.Option(
        2, help="Number of processes per GPU"
    ),
    seed: int = typer.Option(
        27, help="Number of processes per GPU"
    ),
):
    setup_deterministic_environment()
    create_training_dataset_parallel(
        input_dir, output_path, seed, resume, model_path, duration, processes_per_gpu,
    )


if __name__ == "__main__":
    app()
