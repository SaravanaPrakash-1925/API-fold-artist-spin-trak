def log_influence_tensorboard(results_dict, writer, step, top_k=5):
    influences = results_dict.get("top_influences", [])

    if not influences:
        return

    # Histogram
    all_scores = torch.tensor([r["percentage"] for r in influences])
    writer.add_histogram("Influence/AllPercentages", all_scores, global_step=step)

    # chart for Top-k influeces
    top_k_influences = sorted(influences, key=lambda r: r["rank"])[:top_k]
    scalar_dict = {r["track"]: r["percentage"] for r in top_k_influences}
    writer.add_scalars("Influence/TopKPercentage", scalar_dict, global_step=step)