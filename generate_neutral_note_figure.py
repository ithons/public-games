"""Generate the neutral note ablation figure."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    results_dir = Path("results")
    summary_file = results_dir / "neutral_ablation_summary.json"
    
    with open(summary_file, "r") as f:
        summary = json.load(f)
    
    conditions_data = summary["conditions"]
    
    labels = ["None", "Hybrid-Neutral", "Hybrid"]
    condition_keys = ["none", "hybrid_neutral", "hybrid"]
    
    means = [conditions_data[k]["welfare_mean"] for k in condition_keys]
    ci_lows = [conditions_data[k]["welfare_ci_low"] for k in condition_keys]
    ci_highs = [conditions_data[k]["welfare_ci_high"] for k in condition_keys]
    
    errors_low = [m - l for m, l in zip(means, ci_lows)]
    errors_high = [h - m for h, m in zip(ci_highs, means)]
    errors = [errors_low, errors_high]
    
    colors = ["#6c757d", "#ffc107", "#28a745"]  # Gray, Yellow, Green
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(labels, means, yerr=errors, capsize=5, 
                  color=colors, edgecolor="black", linewidth=1.2)
    
    ax.set_ylabel("Total Welfare per Episode", fontsize=12)
    ax.set_xlabel("Memory Condition", fontsize=12)
    ax.set_title("Neutral vs Cooperative Strategy Note", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 260)
    
    ax.axhline(y=240, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Theoretical Maximum")
    ax.axhline(y=120, color="gray", linestyle="--", alpha=0.5, linewidth=1.5, label="Baseline (~50% cooperation)")
    
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f"{mean:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    
    output_paths = [
        Path("blog/assets/neutral_note_ablation.png"),
        Path("analysis/figures/neutral_note_ablation.png"),
    ]
    
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved to {output_path}")
    
    plt.close()
    
    print("\nResults summary:")
    for label, key in zip(labels, condition_keys):
        data = conditions_data[key]
        print(f"  {label}: {data['welfare_mean']:.1f} [{data['welfare_ci_low']:.1f}, {data['welfare_ci_high']:.1f}]")


if __name__ == "__main__":
    main()
