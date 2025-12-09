import logging

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from wut_spintrak.main import get_gradient_embeddings, log_influence_tensorboard
from spintrak import SPINTRAK, show_influence_percentages

logger = logging.getLogger()


def test_spintrak_consistency(
    model: nn.Module,
    prompt: str,
    train_embeddings: torch.Tensor,
    song_names: list[str],
    audio_duration: int = 4,
    num_runs: int = 5,
    top_k: int = 10,
    lambda_: float = 10,
) -> dict:
    """
    Runs the DeterministicSPINTRAK pipeline multiple times on the same input
    and checks if the results are identical across runs.
    """
    spintrak = SPINTRAK(device=model.device)
    all_indices = []
    all_scores = []

    writer = SummaryWriter("runs/my_experiment")
    for run in range(num_runs):
        logger.info(f"🔍 Consistency test run {run + 1}/{num_runs}")
        _, gradients = get_gradient_embeddings(
            model,
            prompt=prompt,
            output_duration=audio_duration,
        )
        attribution_scores = spintrak.get_attribution_scores(
            train_embeddings,
            gradients,
            lambda_=lambda_,
        )
        results = show_influence_percentages(attribution_scores, song_names, top_k)
        indices = [item["sample_index"] for item in results["top_influences"]]
        scores = [
            round(item["similarity_score"], 8) for item in results["top_influences"]
        ]
        all_indices.append(indices)
        all_scores.append(scores)
        log_influence_tensorboard(results, writer, 1)
    writer.close()
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
        "first_run_scores": first_scores,
    }
