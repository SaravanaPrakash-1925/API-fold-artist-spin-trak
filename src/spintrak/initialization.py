import logging
from pathlib import Path

import numpy as np
import torch

from fold_audiocraft.models import MusicGen

DEFAULT_SEED: int = 44


def setup_deterministic_environment(seed: int = DEFAULT_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info("Deterministic environment configured")


def load_model(model_dir=None, prompt=None, audio_input=None):
    if prompt == "":
        prompt = None
    if prompt is None and audio_input is None:
        raise ValueError("You must provide at least a prompt or an audio input.")
    if audio_input is not None:
        model = MusicGen.get_pretrained("facebook/musicgen-melody")
    else:
        model = MusicGen.get_pretrained(Path(model_dir) / "fb-large/")
    return model
