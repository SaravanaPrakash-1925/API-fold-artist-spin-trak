#!/usr/bin/env python
from pathlib import Path

import pandas as pd
import seaborn as sns
import seaborn.objects as so
import torch
import typer

from spintrak.spin_trak import SPINTRAK


def load_test_cases(path: Path) -> tuple[torch.Tensor, list[str], list[int]]:
    test_cases = torch.load(path, weights_only=False)
    seeds = [x["seed"] for x in test_cases]
    names = [x["prompt"] for x in test_cases]
    test_cases = (
        torch.cat([x["gradients"] for x in test_cases]).to("cuda:0").to(torch.float64)
    )
    return test_cases, names, seeds


def load_training_data(gradients_path):
    training_raw = torch.load(gradients_path, weights_only=False)
    if training_raw[0]["name"].startswith("Willie"):
        training_raw = training_raw[::-1]
    training = (
        torch.cat([x["gradients"] for x in training_raw]).to("cuda:0").to(torch.float64)
    )
    return training


def main(
    generated_gradients_file: Path,
    training_gradients_file: Path,
    output_filename: str = "experiment_3_results.svg",
):
    """
    Generate plot with results of the experiment 3.
    """
    test_cases, names, seeds = load_test_cases(generated_gradients_file)
    training = load_training_data(training_gradients_file)
    spintrak = SPINTRAK(training.device)
    scores = spintrak.get_attribution_scores(training, test_cases, lambda_=1).cpu()
    df = pd.DataFrame(
        {
            "Generated sample": names,
            "Random seed": seeds,
            "Rock": scores[:, 0],
            "Salsa": scores[:, 1],
        }
    )
    replace_dict = {
        "Lively salsa groove with bright horns, piano montuno, congas and timbales, rolling bass, and punchy breaks. No vocals.": "Salsa",
        "Moody alternative rock ballad with spacious electric guitars, steady drums, and warm bass; builds from soft verses to a bigger, emotive chorus. No vocals.": "Rock",
    }
    df.replace(
        {"Generated sample": replace_dict},
        inplace=True,
    )
    df = pd.melt(
        df,
        id_vars=["Generated sample", "Random seed"],
        value_vars=["Rock", "Salsa"],
        var_name="Training sample",
        value_name="Attribution score",
    )
    (
        so.Plot(
            df,
            x="Attribution score",
            y="Generated sample",
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
