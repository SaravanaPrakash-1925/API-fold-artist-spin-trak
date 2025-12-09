#fix7

import os
import torch
import torch.nn as nn
import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from tqdm import tqdm
import tempfile
from pathlib import Path
import logging
import urllib.parse
import requests
import numpy as np  # New

def setup_deterministic_environment():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.preferred_linalg_library("cusolver")
        except:
            try:
                torch.backends.cuda.preferred_linalg_library("magma")
            except:
                pass
    logger.info("✅ Deterministic environment configured")

class DeterministicAudioProcessor:
    """Deterministic audio processor for SPINTRAK"""

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512,
                 n_mels=128, seq_len=128, embed_dim=2048, device="cuda"):
        self.sample_rate = sample_rate
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.device = device

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        ).to(device)

        if n_mels != embed_dim:
            setup_deterministic_environment()  # ensure seed before weight init
            self.projector = nn.Linear(n_mels, embed_dim).to(device)
            with torch.no_grad():
                nn.init.xavier_uniform_(self.projector.weight, gain=1.0)
                nn.init.zeros_(self.projector.bias)

        else:
            self.projector = None

    def process_audio_file(self, audio_path: str) -> Optional[torch.Tensor]:
        setup_deterministic_environment()  # Always set seeds before processing
        try:
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.to(self.device)

            # Deterministic normalization
            flat = waveform.flatten()
            mean = torch.sum(flat) / flat.numel()
            var = torch.sum((flat - mean) ** 2) / flat.numel()
            std = torch.sqrt(var + 1e-8)
            waveform = (waveform - mean) / std

            # Mel spectrogram
            spec = self.mel_transform(waveform)
            spec = torchaudio.transforms.AmplitudeToDB()(spec).squeeze(0)

            # Padding/truncation
            if spec.shape[1] > self.seq_len:
                spec = spec[:, :self.seq_len]
            elif spec.shape[1] < self.seq_len:
                pad_size = self.seq_len - spec.shape[1]
                pad = torch.zeros(spec.shape[0], pad_size, device=self.device, dtype=spec.dtype)
                spec = torch.cat([spec, pad], dim=1)

            spec = spec.T  # (seq_len, n_mels)

            if self.projector:
                spec = self.projector(spec)  # (seq_len, embed_dim)

            return spec

        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            return None


# Create log directory
log_dir = "/home/ubuntu/logs"
os.makedirs(log_dir, exist_ok=True)
os.chmod(log_dir, 0o755)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir, "api.log"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
SEQ_LEN = 128
EMBED_DIM = 2048  # Updated to match MusicGen Stereo Large

# Set CUDA backend preference to avoid CUSOLVER issues
if torch.cuda.is_available():
    try:
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except:
        try:
            torch.backends.cuda.preferred_linalg_library("magma")
        except:
            logger.warning("Could not set preferred linear algebra library, using default")

# Dataset and model configurations
MODEL_CONFIGS = {
    "dora:stereo-large-musicgen:fold-ambient": {
        "dataset_path": "/home/ubuntu/ambient",
        "checkpoint_path": "/home/ubuntu/musicgen_finetunes/Ambient/weights.pt",
        "embedding_path": "/home/ubuntu/ambient_audio_embeddings.pt"
    },
    "dora:stereo-large-musicgen:fold-blues": {
        "dataset_path": "/home/ubuntu/blues",
        "checkpoint_path": "/home/ubuntu/musicgen_finetunes/Blues/weights.pt",
        "embedding_path": "/home/ubuntu/blues_audio_embeddings.pt"
    },
    "dora:stereo-large-musicgen:fold-classical": {
        "dataset_path": "/home/ubuntu/classical",
        "checkpoint_path": "/home/ubuntu/musicgen_finetunes/Classical/weights.pt",
        "embedding_path": "/home/ubuntu/classical_audio_embeddings.pt"
    },
    "dora:stereo-large-musicgen:fold-indie-rock": {
        "dataset_path": "/home/ubuntu/indie-rock",
        "checkpoint_path": "/home/ubuntu/musicgen_finetunes/Indie-Rock/weights.pt",
        "embedding_path": "/home/ubuntu/indie-rock_audio_embeddings.pt"
    },
    "dora:stereo-large-musicgen:fold-hip-hop": {
        "dataset_path": "/home/ubuntu/hip-hop",
        "checkpoint_path": "/home/ubuntu/musicgen_finetunes/Hip-Hop/weights.pt",
        "embedding_path": "/home/ubuntu/hip-hop_audio_embeddings.pt"
    },
    "dora:stereo-large-musicgen:fold-reggae": {
        "dataset_path": "/home/ubuntu/reggae",
        "checkpoint_path": "/home/ubuntu/musicgen_finetunes/Reggae/weights.pt",
        "embedding_path": "/home/ubuntu/reggae_audio_embeddings.pt"
    }
}

# Define TransformerDecoder class
class TransformerDecoder(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=2048, num_heads=16, num_layers=6, d_ff=8192):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_projection = nn.Linear(input_dim, embed_dim)  # Fallback for dimension mismatches
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.embedding = nn.Linear(embed_dim, embed_dim)
        self.memory_proj = nn.Linear(embed_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, tgt, memory):
        tgt = self.input_projection(tgt)  # (B, S, input_dim) -> (B, S, embed_dim)
        memory = self.input_projection(memory)  # (B, M, input_dim) -> (B, M, embed_dim)
        tgt = self.embedding(tgt)
        memory = self.memory_proj(memory)
        assert tgt.size(0) == memory.size(0), "Batch sizes must match"
        output = self.transformer(tgt, memory)
        return self.output_layer(output)

# Audio processing functions
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
).to(device)

# def process_audio_file(audio_path: str) -> Optional[torch.Tensor]:
#     logger.info(f"Processing audio file: {audio_path}")
#     try:
#         waveform, sr = torchaudio.load(audio_path)
#         if sr != SAMPLE_RATE:
#             waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)
#         waveform = waveform.to(device)
#         waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
#         spec = mel_transform(waveform)
#         spec = torchaudio.transforms.AmplitudeToDB()(spec)
#         spec = spec.squeeze(0)
#         if spec.shape[1] > SEQ_LEN:
#             spec = spec[:, :SEQ_LEN]
#         elif spec.shape[1] < SEQ_LEN:
#             pad = torch.zeros(spec.shape[0], SEQ_LEN - spec.shape[1]).to(device)
#             spec = torch.cat([spec, pad], dim=1)
#         spec = spec.T  # (seq_len, n_mels) = (128, 128)
#         if N_MELS != EMBED_DIM:
#             projector = nn.Linear(N_MELS, EMBED_DIM).to(device)
#             spec = projector(spec)  # (128, 2048)
#         return spec
#     except Exception as e:
#         logger.error(f"Error processing audio {audio_path}: {e}")
#         return None

def download_file_from_url(url: str, temp_dir: Path) -> str:
    logger.info(f"Downloading file from URL: {url}")
    try:
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme != 'https':
            raise ValueError("URL must be an HTTPS URL")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir)
        temp_file_path = temp_file.name
        temp_file.close()
        
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to download file: HTTP {response.status_code}")
        
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Successfully downloaded file to {temp_file_path}")
        return temp_file_path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")

def get_gradient_embedding(model: nn.Module, audio_tensor: torch.Tensor) -> torch.Tensor:
    """
    Deterministic gradient embedding computation
    """
    setup_deterministic_environment()  # Ensure fixed seeds and ops

    model.eval()
    model.zero_grad()  # Clear previous gradients if any

    if audio_tensor.dim() == 2:
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

    memory = torch.zeros(1, SEQ_LEN, EMBED_DIM, device=audio_tensor.device)

    with torch.enable_grad():
        output = model(audio_tensor, memory)

        # Replace mean() with deterministic sum
        loss = torch.sum(output) / output.numel()

    model.zero_grad()
    loss.backward()

    grad_tensors = []
    for param in model.parameters():
        if param.grad is not None:
            grad_tensors.append(param.grad.view(-1))

    if not grad_tensors:
        raise ValueError("No gradients found during backprop")

    grad_emb = torch.cat(grad_tensors)

    if grad_emb.numel() > EMBED_DIM:
        grad_emb = grad_emb[:EMBED_DIM]
    elif grad_emb.numel() < EMBED_DIM:
        pad = torch.zeros(EMBED_DIM - grad_emb.numel(), device=grad_emb.device)
        grad_emb = torch.cat([grad_emb, pad])

    return grad_emb.unsqueeze(0)  # Return with batch dimension




def compute_influence_scores_robust(train_embeddings: torch.Tensor, generated_embedding: torch.Tensor) -> torch.Tensor:
    """
    Robust version of influence score computation with multiple fallback methods
    """
    logger.info("Computing influence scores with robust method")
    
    if train_embeddings.size(-1) != EMBED_DIM or generated_embedding.size(-1) != EMBED_DIM:
        raise ValueError(f"Embedding dimensions must be {EMBED_DIM}, got train={train_embeddings.size(-1)}, generated={generated_embedding.size(-1)}")
    
    # Method 1: Try original CUSOLVER approach
    try:
        logger.info("Attempting CUSOLVER method")
        return compute_influence_scores_cusolver(train_embeddings, generated_embedding)
    except Exception as e:
        logger.warning(f"CUSOLVER method failed: {e}")
    
    # Method 2: Try CPU computation
    try:
        logger.info("Attempting CPU computation method")
        return compute_influence_scores_cpu(train_embeddings, generated_embedding)
    except Exception as e:
        logger.warning(f"CPU method failed: {e}")
    
    # Method 3: Try alternative GPU method with different backend
    try:
        logger.info("Attempting alternative GPU method")
        return compute_influence_scores_alternative(train_embeddings, generated_embedding)
    except Exception as e:
        logger.warning(f"Alternative GPU method failed: {e}")
    
    # Method 4: Fallback to simple cosine similarity
    logger.info("Using fallback cosine similarity method")
    return compute_influence_scores_cosine(train_embeddings, generated_embedding)

def compute_influence_scores_cusolver(train_embeddings: torch.Tensor, generated_embedding: torch.Tensor) -> torch.Tensor:
    """Original CUSOLVER-based method"""
    K_train = torch.matmul(train_embeddings, train_embeddings.T)
    train_norms = torch.norm(train_embeddings, dim=1, keepdim=True)
    if torch.any(train_norms == 0):
        raise ValueError("Zero norm detected in train_embeddings")
    K_train = K_train / (train_norms @ train_norms.T)
    K_gen = torch.matmul(train_embeddings, generated_embedding.unsqueeze(1)).squeeze(1)
    gen_norm = torch.norm(generated_embedding)
    if gen_norm == 0:
        raise ValueError("Zero norm detected in generated_embedding")
    K_gen = K_gen / (train_norms.squeeze(1) * gen_norm)
    reg = 1e-3 * torch.mean(torch.diag(K_train))
    L = torch.linalg.cholesky(K_train + reg * torch.eye(K_train.shape[0], device=K_train.device))
    y = torch.linalg.solve_triangular(L, K_gen.unsqueeze(1), upper=False)
    influence_scores = torch.linalg.solve_triangular(L.T, y, upper=True).squeeze(1)
    return influence_scores

def compute_influence_scores_cpu(train_embeddings: torch.Tensor, generated_embedding: torch.Tensor) -> torch.Tensor:
    """CPU-based computation to avoid CUDA issues"""
    # Move to CPU for computation
    train_cpu = train_embeddings.cpu()
    gen_cpu = generated_embedding.cpu()
    
    K_train = torch.matmul(train_cpu, train_cpu.T)
    train_norms = torch.norm(train_cpu, dim=1, keepdim=True)
    if torch.any(train_norms == 0):
        raise ValueError("Zero norm detected in train_embeddings")
    K_train = K_train / (train_norms @ train_norms.T)
    K_gen = torch.matmul(train_cpu, gen_cpu.unsqueeze(1)).squeeze(1)
    gen_norm = torch.norm(gen_cpu)
    if gen_norm == 0:
        raise ValueError("Zero norm detected in generated_embedding")
    K_gen = K_gen / (train_norms.squeeze(1) * gen_norm)
    reg = 1e-3 * torch.mean(torch.diag(K_train))
    
    # Use CPU for linear algebra operations
    L = torch.linalg.cholesky(K_train + reg * torch.eye(K_train.shape[0]))
    y = torch.linalg.solve_triangular(L, K_gen.unsqueeze(1), upper=False)
    influence_scores = torch.linalg.solve_triangular(L.T, y, upper=True).squeeze(1)
    
    # Move back to original device
    return influence_scores.to(train_embeddings.device)

def compute_influence_scores_alternative(train_embeddings: torch.Tensor, generated_embedding: torch.Tensor) -> torch.Tensor:
    """Alternative GPU method using pseudoinverse instead of Cholesky"""
    K_train = torch.matmul(train_embeddings, train_embeddings.T)
    train_norms = torch.norm(train_embeddings, dim=1, keepdim=True)
    if torch.any(train_norms == 0):
        raise ValueError("Zero norm detected in train_embeddings")
    K_train = K_train / (train_norms @ train_norms.T)
    K_gen = torch.matmul(train_embeddings, generated_embedding.unsqueeze(1)).squeeze(1)
    gen_norm = torch.norm(generated_embedding)
    if gen_norm == 0:
        raise ValueError("Zero norm detected in generated_embedding")
    K_gen = K_gen / (train_norms.squeeze(1) * gen_norm)
    
    # Use regularized pseudoinverse instead of Cholesky
    reg = 1e-3 * torch.mean(torch.diag(K_train))
    K_reg = K_train + reg * torch.eye(K_train.shape[0], device=K_train.device)
    
    # Use pseudoinverse which is more stable
    influence_scores = torch.matmul(torch.pinverse(K_reg), K_gen)
    return influence_scores

def compute_influence_scores_cosine(train_embeddings: torch.Tensor, generated_embedding: torch.Tensor) -> torch.Tensor:
    """Fallback method using simple cosine similarity"""
    logger.info("Using cosine similarity as fallback")
    
    # Normalize embeddings
    train_norm = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)
    gen_norm = torch.nn.functional.normalize(generated_embedding, p=2, dim=0)
    
    # Compute cosine similarity
    similarity_scores = torch.matmul(train_norm, gen_norm)
    
    return similarity_scores

# Wrapper function to maintain original API
# def compute_influence_scores(train_embeddings: torch.Tensor, generated_embedding: torch.Tensor) -> torch.Tensor:
#     return compute_influence_scores_robust(train_embeddings, generated_embedding)
def compute_influence_scores(train_embeddings: torch.Tensor, generated_embedding: torch.Tensor) -> torch.Tensor:
    return compute_influence_scores_deterministic(train_embeddings, generated_embedding, method="cpu_stable")


def get_song_names(dataset_path: str) -> List[str]:
    logger.info(f"Loading song names from {dataset_path}")
    audio_files = []
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    return sorted(audio_files)

def show_influence_percentages(influence_scores: torch.Tensor, song_names: Optional[List[str]] = None, top_k: int = 10) -> List[Dict]:
    logger.info(f"Computing top {top_k} most influential samples (positive and negative)")
    total_influence = torch.sum(torch.abs(influence_scores))
    if total_influence == 0:
        logger.warning("Total influence is zero, cannot compute percentages")
        return []
    percentages = (influence_scores / total_influence) * 100
    sorted_indices = torch.argsort(torch.abs(percentages), descending=True)
    top_k = min(top_k, len(percentages))
    top_indices = sorted_indices[:top_k]
    results = []
    for i, idx in enumerate(top_indices):
        percent = percentages[idx]
        result = {
            "rank": i + 1,
            "percentage": percent.item(),
            "direction": "positively" if percent.item() >= 0 else "negatively",
            "sample_index": idx.item()
        }
        if song_names is not None and idx.item() < len(song_names):
            result["track"] = os.path.basename(song_names[idx.item()])
        else:
            result["track"] = f"Sample {idx.item()}"
        results.append(result)
    return results

class DeterministicSPINTRAK:
    def __init__(self, device=device, sample_rate=SAMPLE_RATE, embed_dim=EMBED_DIM, seq_len=SEQ_LEN):
        self.device = device
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.audio_processor = DeterministicAudioProcessor(
            sample_rate=sample_rate,
            embed_dim=embed_dim,
            seq_len=seq_len,
            device=device
        )

    def compute_influence_scores(
        self,
        model: nn.Module,
        audio_path: str,
        train_embeddings: torch.Tensor,
        song_names: List[str],
        top_k: int = 10
    ) -> Dict:
        # Step 1 – Make environment deterministic
        setup_deterministic_environment()

        results = {}

        # Step 2 – Process audio deterministically
        audio_tensor = self.audio_processor.process_audio_file(audio_path)
        if audio_tensor is None:
            raise ValueError("Audio processing failed")
        results["audio_shape"] = audio_tensor.shape

        # Step 3 – Compute deterministic gradient embedding
        generated_embedding = get_gradient_embedding(model, audio_tensor).squeeze(0)
        results["embedding_shape"] = generated_embedding.shape

        # Step 4 – Compute deterministic influence scores
        influence_scores = compute_influence_scores_deterministic(
            train_embeddings,
            generated_embedding,
            method="cpu_stable"
        )

        # Step 5 – Get Top-K deterministically
        top_scores, top_indices = self.get_top_k_deterministic(influence_scores, k=top_k)

        top_influences = []
        for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
            top_influences.append({
                "rank": i + 1,
                "similarity_score": score.item(),
                "sample_index": idx.item(),
                "track": os.path.basename(song_names[idx]) if idx < len(song_names) else f"Sample {idx}",
            })

        # Step 6 – Add percentages and directions
        percentage_influences = show_influence_percentages(influence_scores, song_names, top_k)
        for t, perc in zip(top_influences, percentage_influences):
            t["percentage"] = perc["percentage"]
            t["direction"] = perc["direction"]

        results["top_influences"] = top_influences
        return results

    @staticmethod
    def get_top_k_deterministic(influence_scores: torch.Tensor, k: int = 10):
        abs_scores = torch.abs(influence_scores)
        values, indices = torch.sort(abs_scores, descending=True, stable=True)
        top_k = min(k, len(values))
        top_indices = indices[:top_k]
        top_scores = influence_scores[top_indices]
        return top_scores, top_indices
    

def test_spintrak_consistency(model: nn.Module,
                              audio_path: str,
                              train_embeddings: torch.Tensor,
                              song_names: List[str],
                              num_runs: int = 5,
                              top_k: int = 10) -> Dict:
    """
    Runs the DeterministicSPINTRAK pipeline multiple times on the same input
    and checks if the results are identical across runs.
    """
    spintrak = DeterministicSPINTRAK(device=device)
    all_indices = []
    all_scores = []

    for run in range(num_runs):
        logger.info(f"🔍 Consistency test run {run+1}/{num_runs}")
        results = spintrak.compute_influence_scores(
            model=model,
            audio_path=audio_path,
            train_embeddings=train_embeddings,
            song_names=song_names,
            top_k=top_k
        )
        indices = [item["sample_index"] for item in results["top_influences"]]
        scores = [round(item["similarity_score"], 8) for item in results["top_influences"]]
        all_indices.append(indices)
        all_scores.append(scores)

    # Compare all results to the first run
    first_indices = all_indices[0]
    first_scores = all_scores[0]

    indices_consistent = all(lst == first_indices for lst in all_indices)
    scores_consistent = all(lst == first_scores for lst in all_scores)

    return {
        "consistent": indices_consistent and scores_consistent,
        "indices_consistent": indices_consistent,
        "scores_consistent": scores_consistent,
        "num_runs": num_runs,
        "first_run_indices": first_indices,
        "first_run_scores": first_scores
    }


# FastAPI app
app = FastAPI(title="Audio Influence Score API")

# Global storage
MODELS: Dict[str, nn.Module] = {}
EMBEDDINGS: Dict[str, torch.Tensor] = {}
SONG_NAMES: Dict[str, List[str]] = {}

# Response model
class SimilarTrack(BaseModel):
    rank: int
    similarity_score: float
    sample_index: int
    track: str
    percentage: float
    direction: str

class InfluenceScoreResponse(BaseModel):
    message: str
    genre: str
    top_similar_tracks: List[SimilarTrack]

class HealthCheckResponse(BaseModel):
    status: str
    loaded_models: List[str]
    available_models: List[str]
    model_details: Dict

class InfluenceScoreRequest(BaseModel):
    generated_music_url: str
    genre: str

# Custom middleware
@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    logger.info(f"Received request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Completed request: {request.method} {request.url} - Status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error in request: {request.method} {request.url} - {str(e)}")
        return JSONResponse(status_code=500, content={"detail": f"Middleware error: {str(e)}"})

@app.on_event("startup")
def startup_event():
    setup_deterministic_environment()
    print("=" * 60)
    print("🚀 STARTING UP AUDIO INFLUENCE SCORE API")
    print("=" * 60)
    
    logger.info("Starting up API, loading resources")
    print(f"📊 Available MODEL_CONFIGS: {list(MODEL_CONFIGS.keys())}")
    print(f"🖥️  Using device: {device}")
    
    # Log CUDA linear algebra backend info
    if torch.cuda.is_available():
        try:
            current_backend = torch.backends.cuda.preferred_linalg_library()
            print(f"🔧 CUDA Linear Algebra Backend: {current_backend}")
            logger.info(f"CUDA Linear Algebra Backend: {current_backend}")
        except:
            print("🔧 CUDA Linear Algebra Backend: Default")
            logger.info("CUDA Linear Algebra Backend: Default")
    
    logger.info(f"Available MODEL_CONFIGS: {list(MODEL_CONFIGS.keys())}")
    logger.info(f"Using device: {device}")
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n🔄 Loading resources for model: {model_name}")
        print("-" * 40)
        
        logger.info(f"Loading resources for model: {model_name}")
        try:
            print(f"📁 Checking checkpoint: {config['checkpoint_path']}")
            if not os.path.exists(config["checkpoint_path"]):
                print(f"❌ Checkpoint file not found: {config['checkpoint_path']}")
                logger.error(f"Checkpoint file not found: {config['checkpoint_path']}")
                continue
            print("✅ Checkpoint file found")
                
            print(f"📁 Checking embeddings: {config['embedding_path']}")
            if not os.path.exists(config["embedding_path"]):
                print(f"❌ Embedding file not found: {config['embedding_path']}")
                logger.error(f"Embedding file not found: {config['embedding_path']}")
                continue
            print("✅ Embedding file found")
                
            print(f"📁 Checking dataset: {config['dataset_path']}")
            if not os.path.exists(config["dataset_path"]):
                print(f"❌ Dataset path not found: {config['dataset_path']}")
                logger.error(f"Dataset path not found: {config['dataset_path']}")
                continue
            print("✅ Dataset path found")
            
            print("🤖 Loading model...")
            model = TransformerDecoder(input_dim=2048, embed_dim=2048, d_ff=8192).to(device)
            checkpoint = torch.load(config["checkpoint_path"], map_location=device)
            
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint, strict=False)
                print("✅ Loaded checkpoint as dict")
                logger.info(f"Loaded checkpoint as dict for {model_name}")
            elif isinstance(checkpoint, (list, tuple)):
                model_params = list(model.parameters())
                for model_param, loaded_weight in zip(model_params, checkpoint):
                    model_param.copy_(loaded_weight)
                print("✅ Loaded checkpoint as list/tuple")
                logger.info(f"Loaded checkpoint as list/tuple for {model_name}")
            elif isinstance(checkpoint, torch.Tensor):
                ptr = 0
                for param in model.parameters():
                    num_weights = param.numel()
                    param.data.copy_(checkpoint[ptr:ptr + num_weights].view_as(param))
                    ptr += num_weights
                print("✅ Loaded checkpoint as tensor")
                logger.info(f"Loaded checkpoint as tensor for {model_name}")
            else:
                print(f"❌ Unsupported checkpoint format: {type(checkpoint)}")
                logger.error(f"Unsupported checkpoint format for {model_name}: {type(checkpoint)}")
                continue
                
            model.eval()
            MODELS[model_name] = model
            print(f"✅ Successfully loaded model for {model_name}")
            logger.info(f"✓ Successfully loaded model for {model_name}")
            
            print("📊 Loading embeddings...")
            embeddings = torch.load(config["embedding_path"], map_location=device).to(device)
            if embeddings.size(-1) != EMBED_DIM:
                raise ValueError(f"Loaded embeddings for {model_name} have dimension {embeddings.size(-1)}, expected {EMBED_DIM}")
            EMBEDDINGS[model_name] = embeddings
            print(f"✅ Loaded embeddings for {model_name}, shape: {embeddings.shape}")
            logger.info(f"✓ Loaded embeddings for {model_name}, shape: {embeddings.shape}")
            
            print("🎵 Loading song names...")
            song_names = get_song_names(config["dataset_path"])
            if len(song_names) != embeddings.shape[0]:
                print(f"⚠️  Mismatch for {model_name}: {len(song_names)} songs vs {embeddings.shape[0]} embeddings")
                logger.warning(f"Mismatch for {model_name}: {len(song_names)} songs vs {embeddings.shape[0]} embeddings")
                song_names = [f"Sample {i}" for i in range(embeddings.shape[0])]
            SONG_NAMES[model_name] = song_names
            print(f"✅ Loaded {len(song_names)} song names for {model_name}")
            logger.info(f"✓ Loaded {len(song_names)} song names for {model_name}")
            
        except Exception as e:
            print(f"❌ Error loading resources for {model_name}: {e}")
            logger.error(f"✗ Error loading resources for {model_name}: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            continue
    
    print("\n" + "=" * 60)
    print("📋 STARTUP SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully loaded models: {list(MODELS.keys())}")
    print(f"📊 Available embeddings: {list(EMBEDDINGS.keys())}")
    print(f"🎵 Available song collections: {list(SONG_NAMES.keys())}")
    
    logger.info(f"Startup complete. Successfully loaded models: {list(MODELS.keys())}")
    logger.info(f"Available embeddings: {list(EMBEDDINGS.keys())}")
    logger.info(f"Available song collections: {list(SONG_NAMES.keys())}")
    
    if not MODELS:
        print("⚠️  WARNING: No models were loaded successfully!")
        logger.error("⚠️  WARNING: No models were loaded successfully!")
    else:
        print(f"🎉 Total models loaded: {len(MODELS)}")
        logger.info(f"✓ Total models loaded: {len(MODELS)}")
    
    print("=" * 60)
    print("🚀 API STARTUP COMPLETE!")
    print("=" * 60)

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    logger.info("Health check requested")
    
    model_details = {}
    for model_name in MODEL_CONFIGS.keys():
        model_details[model_name] = {
            "model_loaded": model_name in MODELS,
            "embeddings_loaded": model_name in EMBEDDINGS,
            "songs_loaded": model_name in SONG_NAMES,
            "embedding_shape": str(EMBEDDINGS[model_name].shape) if model_name in EMBEDDINGS else None,
            "num_songs": len(SONG_NAMES[model_name]) if model_name in SONG_NAMES else None
        }
    
    status = "healthy" if MODELS else "unhealthy"
    
    return HealthCheckResponse(
        status=status,
        loaded_models=list(MODELS.keys()),
        available_models=list(MODEL_CONFIGS.keys()),
        model_details=model_details
    )

def compute_influence_scores_deterministic(train_embeddings: torch.Tensor, 
                                           generated_embedding: torch.Tensor,
                                           method: str = "cpu_stable") -> torch.Tensor:
    """
    Compute influence scores with guaranteed deterministic behavior using stable CPU methods.
    """
    setup_deterministic_environment()  # Ensure deterministic environment
    
    if train_embeddings.size(-1) != generated_embedding.size(-1):
        raise ValueError(f"Embedding dimension mismatch: {train_embeddings.size(-1)} vs {generated_embedding.size(-1)}")
    
    if generated_embedding.dim() > 1:
        generated_embedding = generated_embedding.squeeze()
    
    if method == "cpu_stable":
        # Use stable CPU double-precision SVD-based solve
        train_cpu = train_embeddings.detach().cpu().double()
        gen_cpu = generated_embedding.detach().cpu().double()
        
        K_train = torch.matmul(train_cpu, train_cpu.T)
        
        train_norms_sq = torch.sum(train_cpu * train_cpu, dim=1, keepdim=True)
        train_norms = torch.sqrt(train_norms_sq + 1e-12)  # Avoid zero division
        
        K_train = K_train / (train_norms @ train_norms.T)
        
        K_gen = torch.matmul(train_cpu, gen_cpu.unsqueeze(1)).squeeze(1)
        gen_norm = torch.sqrt(torch.sum(gen_cpu * gen_cpu) + 1e-12)
        K_gen = K_gen / (train_norms.squeeze(1) * gen_norm)
        
        reg = 1e-3 * torch.mean(torch.diag(K_train))
        K_reg = K_train + reg * torch.eye(K_train.shape[0], dtype=torch.float64)
        
        try:
            U, S, Vh = torch.linalg.svd(K_reg)
            S_reg = torch.clamp(S, min=1e-10)
            K_inv = (Vh.T / S_reg.unsqueeze(0)) @ U.T
            influence_scores = K_inv @ K_gen
        except Exception:
            # Fallback to pseudoinverse if SVD fails
            influence_scores = torch.pinverse(K_reg) @ K_gen
        
        return influence_scores.float().to(train_embeddings.device)
    
    else:
        raise ValueError(f"Unknown method: {method}")

@app.post("/gradient", response_model=InfluenceScoreResponse)
async def compute_influence_scores_endpoint(request: InfluenceScoreRequest):
    setup_deterministic_environment()
    logger.info(f"Received request for genre: {request.genre}")
    
    # Make genre case-insensitive by normalizing to lowercase
    genre_normalized = request.genre.lower()
    
    # Create a mapping from genre to actual model names
    model_name_mapping = {
        key.split(':')[-1].replace('fold-', '').lower(): key 
        for key in MODEL_CONFIGS.keys()
    }
    
    if genre_normalized not in model_name_mapping:
        logger.error(f"Invalid genre: {request.genre}")
        available_genres = [key.split(':')[-1].replace('fold-', '') for key in MODEL_CONFIGS.keys()]
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid genre '{request.genre}'. Choose from: {available_genres} (case-insensitive)"
        )
    
    # Get the actual model name and genre
    actual_model_name = model_name_mapping[genre_normalized]
    genre = genre_normalized
    
    if actual_model_name not in MODELS or actual_model_name not in EMBEDDINGS:
        logger.error(f"Resources for {actual_model_name} not loaded")
        raise HTTPException(status_code=500, detail=f"Resources for {actual_model_name} not loaded")
    
    upload_dir = Path("/home/ubuntu/uploads")
    upload_dir.mkdir(exist_ok=True)
    os.chmod(str(upload_dir), 0o755)
    
    temp_file_path = None
    try:
        # Download audio file from the provided HTTPS URL
        temp_file_path = download_file_from_url(request.generated_music_url, upload_dir)
        
        # Check file size (limit to 10 MB)
        file_size = os.path.getsize(temp_file_path)
        if file_size > 10 * 1024 * 1024:
            logger.error("File size exceeds 10 MB")
            raise HTTPException(status_code=400, detail="File size exceeds 10 MB")
        
        #generated_audio = process_audio_file(temp_file_path)

        # Add once, globally or per call
        audio_processor = DeterministicAudioProcessor(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            seq_len=SEQ_LEN,
            embed_dim=EMBED_DIM,
            device=device
        )

        # Then use:
        generated_audio = audio_processor.process_audio_file(temp_file_path)

        if generated_audio is None:
            logger.error("Failed to process uploaded audio file")
            raise HTTPException(status_code=400, detail="Failed to process uploaded audio file")
        
        # Check generated_audio dimension
        if generated_audio.size(-1) != EMBED_DIM or generated_audio.size(0) != SEQ_LEN:
            logger.error(f"generated_audio shape ({generated_audio.shape}) does not match expected ({SEQ_LEN}, {EMBED_DIM})")
            raise HTTPException(status_code=400, detail=f"generated_audio shape ({generated_audio.shape}) does not match expected ({SEQ_LEN}, {EMBED_DIM})")
        
        model = MODELS[actual_model_name]
        #generated_embedding = get_gradient_embedding(model, generated_audio).squeeze(0)
        generated_embedding = get_gradient_embedding(model, generated_audio).squeeze(0)

        
        # Check generated_embedding dimension
        if generated_embedding.size(-1) != EMBED_DIM:
            logger.error(f"generated_embedding last dimension ({generated_embedding.size(-1)}) does not match EMBED_DIM ({EMBED_DIM})")
            raise HTTPException(status_code=400, detail=f"generated_embedding last dimension ({generated_embedding.size(-1)}) does not match EMBED_DIM ({EMBED_DIM})")
        
        train_embeddings = EMBEDDINGS[actual_model_name]
        #influence_scores = compute_influence_scores(train_embeddings, generated_embedding)
        influence_scores = compute_influence_scores_deterministic(train_embeddings, generated_embedding, method="cpu_stable")

        song_names = SONG_NAMES[actual_model_name]
        top_k = min(10, len(influence_scores))
        # top_scores, top_indices = torch.topk(influence_scores.abs(), k=top_k)
        def get_top_k_deterministic(influence_scores: torch.Tensor, k: int = 10):
            abs_scores = torch.abs(influence_scores)
            # Stable sort: always same order for ties
            values, indices = torch.sort(abs_scores, descending=True, stable=True)
            top_k = min(k, len(values))
            top_values = values[:top_k]
            top_indices = indices[:top_k]
            # Get the true signed scores for those indices
            top_scores = influence_scores[top_indices]
            return top_scores, top_indices

        # Usage inside your endpoint:
        top_scores, top_indices = get_top_k_deterministic(influence_scores, k=top_k)

        top_influences = []
        for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
            actual_score = influence_scores[idx].item()
            influence = {
                "rank": i + 1,
                "similarity_score": actual_score,
                "sample_index": idx.item(),
                "track": os.path.basename(song_names[idx.item()]) if idx.item() < len(song_names) else f"Sample {idx.item()}"
            }
            top_influences.append(influence)
        
        percentage_influences = show_influence_percentages(influence_scores, song_names, top_k)
        
        logger.info(f"Successfully processed request for {actual_model_name} with top {len(top_influences)} positive influences")

        spintrak = DeterministicSPINTRAK(device=device)
        results = spintrak.compute_influence_scores(
                    model=MODELS[actual_model_name],
                    audio_path=temp_file_path,
                    train_embeddings=EMBEDDINGS[actual_model_name],
                    song_names=SONG_NAMES[actual_model_name],
                    top_k=10
                )

        return InfluenceScoreResponse(
                    message="Audio processed successfully",
                    genre=genre,
                    top_similar_tracks=results["top_influences"]
                )

        # return InfluenceScoreResponse(
        #     message="Audio processed successfully",
        #     genre=genre,
        #     top_similar_tracks=[{**infl, "percentage": perc["percentage"], "direction": perc["direction"]} for infl, perc in zip(top_influences, percentage_influences)]
        # )
    
    except Exception as e:
        logger.error(f"Error processing request for {request.genre}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Deleted temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Failed to delete temporary file {temp_file_path}: {e}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    import uvicorn
    print("🌐 Starting FastAPI server on http://0.0.0.0:8002")
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8002)