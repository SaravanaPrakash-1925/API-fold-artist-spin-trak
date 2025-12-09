#!/usr/bin/env python
from pathlib import Path

import pandas as pd
import seaborn as sns
import seaborn.objects as so
import torch
import typer

from spintrak.spin_trak import SPINTRAK

DURATIONS = [30, 60, 90]


def load_test_cases(path: Path) -> tuple[torch.Tensor, list[str], list[int]]:
    return torch.load(path, weights_only=False)


def load_training_data_duration(gradients_path):
    training_raw = torch.load(gradients_path, weights_only=False)
    if training_raw[0]["name"].startswith("Willie"):
        training_raw = training_raw[::-1]
    training = (
        torch.cat([x["gradients"] for x in training_raw]).to("cuda:0").to(torch.float64)
    )
    return training


def load_all_training_gradients(gradients_paths):
    training_gradients = {}
    for gradient_path in gradients_paths:
        gradients = load_training_data_duration(gradient_path)
        duration = None
        for dur in DURATIONS:
            if str(dur) in gradient_path.name:
                duration = dur
        if not duration:
            print(
                "Training gradient files should have duration in the name (30, 60 or 90)"
            )
        training_gradients[duration] = gradients
    assert len(training_gradients) == 3, (
        "Expected three different durations: 30, 60, 90"
    )
    return training_gradients


def get_scores_df_and_duration_means(test_cases, training_gradients):
    spintrak = SPINTRAK("cuda:0")
    all_seeds = []
    all_durations = []
    all_scores_rock = []
    all_scores_salsa = []
    duration_gradient_means = {}
    for duration in DURATIONS:
        seeds = [x["seed"] for x in test_cases if x["duration"] == duration]
        all_seeds.extend(seeds)
        durations = [str(duration)] * len(seeds)
        all_durations.extend(durations)
        test_gradients = (
            torch.cat([x["gradients"] for x in test_cases if x["duration"] == duration])
            .to("cuda:0")
            .to(torch.float64)
        )
        duration_gradient_means[duration] = test_gradients.mean().item()
        scores = spintrak.get_attribution_scores(
            training_gradients[duration], test_gradients, lambda_=10
        ).cpu()
        all_scores_rock.append(scores[:, 0])
        all_scores_salsa.append(scores[:, 1])
    all_scores_rock = torch.cat(all_scores_rock).flatten()
    all_scores_salsa = torch.cat(all_scores_salsa).flatten()
    df = pd.DataFrame(
        {
            "Duration": all_durations,
            "Random seed": all_seeds,
            "Rock": all_scores_rock,
            "Salsa": all_scores_salsa,
        }
    )
    return df, duration_gradient_means


def main(
    generated_gradients_file: Path,
    training_gradients_files: list[Path],
    output_filename: str = "experiment_1_results.svg",
):
    """
    Generate plot with results of the experiment 1.
    """
    test_cases = load_test_cases(generated_gradients_file)
    training = load_all_training_gradients(training_gradients_files)
    df, duration_gradient_means = get_scores_df_and_duration_means(test_cases, training)
    df_durations = pd.DataFrame(
        {
            "Duration": list(duration_gradient_means),
            "Mean gradient information": list(duration_gradient_means.values()),
        }
    )
    print(df_durations.to_markdown(index=False, tablefmt="grid"))
    df = pd.melt(
        df,
        id_vars=["Duration", "Random seed"],
        value_vars=["Rock", "Salsa"],
        var_name="Training sample",
        value_name="Attribution score",
    )
    (
        so.Plot(
            df,
            x="Attribution score",
            y="Duration",
            color="Training sample",
            marker="Random seed",
        )
        .add(so.Dot(pointsize=7, edgecolor="white"), so.Jitter())
        .theme(
            {
                **sns.axes_style("ticks"),
                "axes.spines.right": False,
                "axes.spines.top": False,
            }
        )
        .save(output_filename, bbox_inches="tight")
    )
    print(f"Saved results file: {output_filename}")


def run():
    typer.run(main)


if __name__ == "__main__":
    run()
