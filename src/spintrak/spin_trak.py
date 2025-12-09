import logging
import os
from typing import Optional

import torch

logger = logging.getLogger()


def show_influence_percentages(
    influence_scores: torch.Tensor,
    song_names: Optional[list[str]] = None,
    top_k: int = 10,
) -> list[dict]:
    logger.info(
        f"Computing top {top_k} most influential samples (positive and negative)"
    )
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
            "sample_index": idx.item(),
        }
        if song_names is not None and idx.item() < len(song_names):
            result["track"] = os.path.basename(song_names[idx.item()])
        else:
            result["track"] = f"Sample {idx.item()}"
        results.append(result)
    return results


class SPINTRAK:
    def __init__(
        self,
        device,
    ):
        self.device = device

    @staticmethod
    def get_top_influences(
        influence_scores: torch.Tensor, song_names: list[str], top_k: int = 10
    ):
        top_scores, top_indices = SPINTRAK.get_top_k_deterministic(
            influence_scores, k=top_k
        )

        top_influences = []
        for i, (score, idx) in enumerate(zip(top_scores, top_indices)):
            top_influences.append(
                {
                    "rank": i + 1,
                    "similarity_score": score.item(),
                    "sample_index": idx.item(),
                    "track": os.path.basename(song_names[idx])
                    if idx < len(song_names)
                    else f"Sample {idx}",
                }
            )

        # Add percentages and directions
        percentage_influences = show_influence_percentages(
            influence_scores, song_names, top_k
        )
        for t, perc in zip(top_influences, percentage_influences):
            t["percentage"] = perc["percentage"]
            t["direction"] = perc["direction"]

        return top_influences

    @torch.no_grad()
    def get_attribution_scores(
        self,
        training_gradient_projections: torch.Tensor,
        generated_gradient_projections: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        logger.info("Starting attribution calculations")
        lambda_ = torch.Tensor([lambda_]).to(self.device)
        kernel = training_gradient_projections.T @ training_gradient_projections
        kernel_scaled = kernel + lambda_ * torch.eye(kernel.shape[0]).to(self.device)
        kernel_scaled = torch.linalg.inv(kernel_scaled)
        scores = (
            generated_gradient_projections @ kernel_scaled
        ) @ training_gradient_projections.T
        logger.info("Attributions calculated")
        return scores

    @torch.no_grad()
    def precompute_training_kernel_tensor(
        self,
        training_gradient_projections: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        lambda_ = torch.Tensor([lambda_]).to(self.device)
        kernel = training_gradient_projections.T @ training_gradient_projections
        kernel_scaled = kernel + lambda_ * torch.eye(kernel.shape[0]).to(self.device)
        kernel_scaled = torch.linalg.inv(kernel_scaled)
        return kernel_scaled @ training_gradient_projections.T

    @torch.no_grad()
    @staticmethod
    def get_attribution_scores_with_precomputed_kernel(
        precomputed: torch.Tensor,
        generated_gradient_projections: torch.Tensor,
    ) -> torch.Tensor:
        scores = generated_gradient_projections @ precomputed
        return scores

    @staticmethod
    def get_top_k_deterministic(influence_scores: torch.Tensor, k: int = 10):
        abs_scores = torch.abs(influence_scores)
        values, indices = torch.sort(abs_scores, descending=True, stable=True)
        top_k = min(k, len(values))
        top_indices = indices[:top_k]
        top_scores = influence_scores[top_indices]
        return top_scores, top_indices
