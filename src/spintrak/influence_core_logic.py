import torch
import numpy as np
from pathlib import Path
import os
from spin_trak import SPINTRAK, show_influence_percentages

def get_song_names_from_training_directory(training_dir):
    """Get sorted song names aligned with embeddings order by filename"""
    directory = Path(training_dir)
    music_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac"}

    # Collect stems of JSON descriptions
    descriptions = {f.stem for f in directory.rglob("*.json")}

    music_files = []
    for file in directory.rglob("*"):
        if (file.is_file() and
            file.suffix.lower() in music_extensions and
            file.stem in descriptions):
            music_files.append(file)

    # Sort files lexicographically by filename to match .pt sorting order
    music_files_sorted = sorted(music_files, key=lambda f: f.name)

    song_names = [str(f) for f in music_files_sorted]  # store full path as str
    return song_names

def load_training_gradients(bin_file_path):
    """Load training gradients - RAW (as SPINTRAK expects)"""
    print(f"\nLoading training gradients from: {bin_file_path}")

    data = np.fromfile(bin_file_path, dtype=np.float32)
    data = torch.from_numpy(data)

    embedding_dim = 8192
    num_samples = data.shape[0] // embedding_dim
    data = data.reshape(num_samples, embedding_dim)

    print(f"Loaded {num_samples} samples with {embedding_dim} dimensions each")
    print("Using RAW gradients (as SPINTRAK expects)")

    return data

def load_generated_gradients(pt_file_path):
    """Load generated gradients - RAW (as SPINTRAK expects)"""
    print(f"\nLoading generated gradients from: {pt_file_path}")

    data = torch.load(pt_file_path, weights_only=False, map_location='cpu')

    if isinstance(data, list):
        if isinstance(data[0], dict) and 'gradients' in data[0]:
            gradients = data[0]['gradients']
        elif isinstance(data[0], torch.Tensor):
            gradients = data[0]
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

    print(f"Extracted gradients shape: {gradients.shape}")
    print("Using RAW gradients (as SPINTRAK expects)")

    return gradients

def display_all_influences(influence_scores, song_names):
    """Display ALL samples with their influence scores and percentages"""

    scores_flat = influence_scores.squeeze(0).cpu()

    print("\n" + "=" * 90)
    print("ALL TRAINING SAMPLES - COMPLETE INFLUENCE BREAKDOWN")
    print("=" * 90)

    total_samples = len(scores_flat)

    all_percentages = show_influence_percentages(
        scores_flat,
        song_names,
        top_k=total_samples  # Get ALL samples
    )

    print(f"\nTotal training samples: {total_samples}")
    print(f"Displaying all {total_samples} samples with scores and percentages:\n")

    for result in all_percentages:
        rank = result['rank']
        track = result['track']
        percentage = result['percentage']
        direction = result['direction']
        sample_idx = result['sample_index']

        actual_score = scores_flat[sample_idx].item()

        print(f"Rank {rank:3d}: {os.path.basename(track)[:65]:<65}")
        print(f"           Index: {sample_idx:3d} | Score: {actual_score:12.6f} | "
              f"Percentage: {percentage:8.4f}% ({direction})")
        print()

    return all_percentages

def save_complete_results(influence_scores, song_names, all_influences, output_prefix="spintrak_complete"):
    """Save complete results for ALL samples"""

    results_dict = {
        'influence_scores': influence_scores.cpu(),
        'song_names': song_names,
        'all_influences': all_influences,
        'method': 'Pure SPINTRAK with complete influence breakdown'
    }

    torch.save(results_dict, f"{output_prefix}.pt")
    print(f"\n✅ Saved: {output_prefix}.pt")

    with open(f"{output_prefix}.txt", "w") as f:
        f.write("=" * 90 + "\n")
        f.write("SPINTRAK COMPLETE INFLUENCE ANALYSIS\n")
        f.write(f"Total samples: {len(all_influences)}\n")
        f.write("=" * 90 + "\n\n")

        for result in all_influences:
            f.write(f"Rank {result['rank']}: {os.path.basename(result['track'])}\n")
            f.write(f"  Sample Index: {result['sample_index']}\n")
            f.write(f"  Percentage: {result['percentage']:.4f}% ({result['direction']})\n\n")

    print(f"✅ Saved: {output_prefix}.txt (contains ALL {len(all_influences)} samples)")

    import csv
    with open(f"{output_prefix}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Song Name', 'Sample Index', 'Percentage', 'Direction'])

        for result in all_influences:
            writer.writerow([
                result['rank'],
                os.path.basename(result['track']),
                result['sample_index'],
                f"{result['percentage']:.4f}%",
                result['direction']
            ])

    print(f"✅ Saved: {output_prefix}.csv (contains ALL {len(all_influences)} samples)")

def print_top20_positive_normalized(influence_scores, song_names=None):
    """Print top 20 positive influences normalized to sum to 100%"""

    scores_flat = influence_scores.squeeze(0).cpu()

    pos_indices = (scores_flat > 0).nonzero(as_tuple=True)[0]
    pos_scores = scores_flat[pos_indices]

    sum_pos = pos_scores.sum().item()
    normalized_percent = (pos_scores * 100) / sum_pos if sum_pos != 0 else torch.zeros_like(pos_scores)

    results = []
    for i, idx in enumerate(pos_indices):
        percent = normalized_percent[i].item()
        score = scores_flat[idx].item()
        name = song_names[idx] if song_names is not None else f"Sample_{idx.item()}"
        results.append({
            'index': idx.item(),
            'score': score,
            'normalized_percentage': percent,
            'name': os.path.basename(name)
        })

    results.sort(key=lambda x: x['normalized_percentage'], reverse=True)

    print("\n" + "=" * 90)
    print("TOP 20 POSITIVELY INFLUENTIAL SAMPLES (Normalized to 100%)")
    print("=" * 90)
    print(f"\nTotal positive samples: {len(results)}")
    print(f"Sum of positive scores: {sum_pos:.2f}")
    print(f"\nTop 20 contributors:\n")

    for rank, r in enumerate(results[:20], 1):
        print(f"Rank {rank:2d}: {r['name']:<50}  Index: {r['index']:3d}")
        print(f"         Score: {r['score']:12.6f} | Normalized: {r['normalized_percentage']:8.4f}%")
        print()

def main():
    print("=" * 90)
    print("SPINTRAK - COMPLETE METHOD WITH ALL SAMPLES")
    print("=" * 90)
    print("\nThis displays:")
    print("  ✓ ALL training samples (not just top 10)")
    print("  ✓ Complete influence scores")
    print("  ✓ Complete percentages (using show_influence_percentages)")
    print("  ✓ Ranked from most to least influential")
    print("  ✓ Top 20 positive influences (normalized)")

    training_bin_path = "/home/ubuntu/poland_teams_codebase/fold-artist-spin-trak/phase1_training_gradients_combined.bin"
    generated_pt_path = "/home/ubuntu/generated_embeddings_1/f8f265ca_7_aa17ecfd_1764168330_track0.pt"
    training_audio_dir = "/home/ubuntu/Phase-1"

    lambda_val = 0.1

    print(f"\nConfiguration:")
    print(f"  Lambda: {lambda_val}")

    print("\n" + "=" * 90)
    print("LOADING DATA")
    print("=" * 90)

    song_names = get_song_names_from_training_directory(training_audio_dir)
    if len(song_names) == 0:
        print("⚠️  No song names found, using generic names")
        song_names = [f"Sample_{i}" for i in range(99)]

    # Verification print using same indexing and basename style as SPINTRAK
    print("\nPhase-1 training dataset song names with their indices:")
    for idx, name in enumerate(song_names):
        print(f"Index {idx:3d}: {os.path.basename(name)}")
    print()

    training_gradients = load_training_gradients(training_bin_path)
    generated_gradients = load_generated_gradients(generated_pt_path)

    if len(song_names) != len(training_gradients):
        print(f"\n⚠️  Count mismatch: {len(song_names)} names vs {len(training_gradients)} gradients")
        if len(song_names) > len(training_gradients):
            song_names = song_names[:len(training_gradients)]
        else:
            song_names.extend([f"Sample_{i}" for i in range(len(song_names), len(training_gradients))])

    print("\n" + "=" * 90)
    print("COMPUTING INFLUENCES")
    print("=" * 90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    training_gradients = training_gradients.to(device)
    generated_gradients = generated_gradients.to(device)

    spintrak = SPINTRAK(device=device)

    print(f"\nComputing influence scores with lambda={lambda_val}...")
    influence_scores = spintrak.get_attribution_scores(
        training_gradients,
        generated_gradients,
        lambda_val
    )

    print(f"Influence scores computed: {influence_scores.shape}")

    all_influences = display_all_influences(influence_scores, song_names)

    print("\n" + "=" * 90)
    print("SUMMARY STATISTICS")
    print("=" * 90)

    percentages = [inf['percentage'] for inf in all_influences]
    positive_pct = [p for p in percentages if p > 0]
    negative_pct = [p for p in percentages if p < 0]

    print(f"\nTotal samples: {len(all_influences)}")
    print(f"Positive influences: {len(positive_pct)} samples")
    print(f"Negative influences: {len(negative_pct)} samples")
    print(f"\nPercentage range:")
    print(f"  Highest: {max(percentages):.4f}%")
    print(f"  Lowest: {min(percentages):.4f}%")
    print(f"  Total (should be ~100%): {sum(abs(p) for p in percentages):.4f}%")

    print_top20_positive_normalized(influence_scores, song_names)

    print("\n" + "=" * 90)
    print("SAVING COMPLETE RESULTS")
    print("=" * 90)

    save_complete_results(influence_scores, song_names, all_influences,
                         output_prefix="spintrak_all_samples")

    print("\n" + "=" * 90)
    print("COMPLETED!")
    print("=" * 90)
    print(f"\nAll {len(all_influences)} samples saved with complete influence breakdown.")

if __name__ == "__main__":
    main()
