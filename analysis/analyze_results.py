"""
Analysis and Plotting Utilities

Load experiment results, compute metrics, and generate figures for the blog.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import seaborn as sns

# Set style for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


@dataclass
class ExperimentMetrics:
    """Computed metrics for a single experiment condition."""
    memory_type: str
    num_episodes: int
    num_rounds: int
    num_agents: int
    
    # Per-round metrics (averaged across episodes)
    mean_contribution_per_round: list[float] = field(default_factory=list)
    std_contribution_per_round: list[float] = field(default_factory=list)
    
    # Episode-level metrics
    total_welfare_mean: float = 0.0
    total_welfare_std: float = 0.0
    final_budget_mean: float = 0.0
    final_budget_std: float = 0.0
    
    # Inequality metrics (Gini coefficient of final budgets)
    gini_mean: float = 0.0
    gini_std: float = 0.0
    
    # Cooperation metrics
    cooperation_rate: float = 0.0  # Fraction of rounds with high cooperation
    cooperation_stability: float = 0.0  # 1 - variance in cooperation
    
    # Cost metrics
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    mean_memory_tokens: float = 0.0
    total_tokens_per_episode: float = 0.0


def gini_coefficient(values: list[float]) -> float:
    """Compute Gini coefficient for inequality measurement."""
    if not values or len(values) < 2:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load all experiment results from a directory."""
    results = {}
    
    for json_file in results_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        memory_type = data["config"]["memory_type"]
        results[memory_type] = data
    
    return results


def compute_metrics(data: dict) -> ExperimentMetrics:
    """Compute aggregate metrics from experiment results."""
    config = data["config"]
    results = data["results"]
    
    num_episodes = len(results)
    num_rounds = config["num_rounds"]
    num_agents = config["num_agents"]
    
    metrics = ExperimentMetrics(
        memory_type=config["memory_type"],
        num_episodes=num_episodes,
        num_rounds=num_rounds,
        num_agents=num_agents,
    )
    
    # Collect per-round contributions across episodes
    round_contributions = [[] for _ in range(num_rounds)]
    
    welfare_values = []
    gini_values = []
    prompt_tokens = []
    completion_tokens = []
    memory_tokens = []
    
    for episode in results:
        contributions = episode["contributions"]
        
        for r, round_contribs in enumerate(contributions):
            if r < num_rounds:
                round_contributions[r].extend(round_contribs)
        
        welfare_values.append(episode["total_welfare"])
        
        # Compute Gini for this episode
        final_budgets = episode["final_budgets"]
        if final_budgets:
            gini_values.append(gini_coefficient(final_budgets))
        
        prompt_tokens.append(episode.get("total_prompt_tokens", 0))
        completion_tokens.append(episode.get("total_completion_tokens", 0))
        
        mem_tokens = episode.get("memory_token_estimates", [])
        if mem_tokens:
            memory_tokens.append(np.mean(mem_tokens))
    
    # Compute per-round statistics
    for r, contribs in enumerate(round_contributions):
        if contribs:
            metrics.mean_contribution_per_round.append(np.mean(contribs))
            metrics.std_contribution_per_round.append(np.std(contribs))
        else:
            metrics.mean_contribution_per_round.append(0)
            metrics.std_contribution_per_round.append(0)
    
    # Episode-level statistics
    if welfare_values:
        metrics.total_welfare_mean = np.mean(welfare_values)
        metrics.total_welfare_std = np.std(welfare_values)
    
    if gini_values:
        metrics.gini_mean = np.mean(gini_values)
        metrics.gini_std = np.std(gini_values)
    
    # Compute cooperation metrics
    all_contributions = []
    for episode in results:
        for round_contribs in episode["contributions"]:
            all_contributions.extend(round_contribs)
    
    if all_contributions:
        max_contrib = config["max_contribution"]
        high_threshold = max_contrib * 0.5
        
        metrics.cooperation_rate = np.mean([c >= high_threshold for c in all_contributions])
        
        per_round_means = metrics.mean_contribution_per_round
        if per_round_means:
            variance = np.var(per_round_means) / (max_contrib ** 2) if max_contrib > 0 else 0
            metrics.cooperation_stability = 1 - min(1, variance * 10)
    
    # Token metrics
    if prompt_tokens:
        metrics.mean_prompt_tokens = np.mean(prompt_tokens)
    if completion_tokens:
        metrics.mean_completion_tokens = np.mean(completion_tokens)
    if memory_tokens:
        metrics.mean_memory_tokens = np.mean(memory_tokens)
    
    metrics.total_tokens_per_episode = metrics.mean_prompt_tokens + metrics.mean_completion_tokens
    
    return metrics


def plot_cooperation_over_time(
    all_metrics: dict[str, ExperimentMetrics],
    output_path: Path,
    title: str = "Cooperation Over Time by Memory Condition",
) -> None:
    """Plot average contribution per round for each memory condition."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(all_metrics))
    
    for (mem_type, metrics), color in zip(all_metrics.items(), colors):
        rounds = list(range(1, len(metrics.mean_contribution_per_round) + 1))
        means = metrics.mean_contribution_per_round
        stds = metrics.std_contribution_per_round
        
        ax.plot(rounds, means, label=mem_type, color=color, linewidth=2, marker="o", markersize=4)
        ax.fill_between(
            rounds,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.2,
            color=color,
        )
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Mean Contribution", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title="Memory Type", loc="best")
    ax.set_xticks(range(1, metrics.num_rounds + 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_welfare_comparison(
    all_metrics: dict[str, ExperimentMetrics],
    output_path: Path,
    title: str = "Total Welfare by Memory Condition",
) -> None:
    """Bar plot comparing total welfare across memory conditions."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mem_types = list(all_metrics.keys())
    means = [m.total_welfare_mean for m in all_metrics.values()]
    stds = [m.total_welfare_std for m in all_metrics.values()]
    
    colors = sns.color_palette("husl", len(mem_types))
    bars = ax.bar(mem_types, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=1)
    
    ax.set_xlabel("Memory Condition", fontsize=12)
    ax.set_ylabel("Total Welfare (sum of payoffs)", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.annotate(
            f"{mean:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_inequality_comparison(
    all_metrics: dict[str, ExperimentMetrics],
    output_path: Path,
    title: str = "Inequality (Gini) by Memory Condition",
) -> None:
    """Bar plot comparing inequality across memory conditions."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mem_types = list(all_metrics.keys())
    means = [m.gini_mean for m in all_metrics.values()]
    stds = [m.gini_std for m in all_metrics.values()]
    
    colors = sns.color_palette("husl", len(mem_types))
    bars = ax.bar(mem_types, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=1)
    
    ax.set_xlabel("Memory Condition", fontsize=12)
    ax.set_ylabel("Gini Coefficient (0=equal, 1=unequal)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1)
    
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cost_vs_performance(
    all_metrics: dict[str, ExperimentMetrics],
    output_path: Path,
    title: str = "Welfare vs Token Cost",
) -> None:
    """Scatter plot of welfare vs token cost."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(all_metrics))
    
    for (mem_type, metrics), color in zip(all_metrics.items(), colors):
        ax.scatter(
            metrics.total_tokens_per_episode,
            metrics.total_welfare_mean,
            s=150,
            c=[color],
            label=mem_type,
            edgecolors="black",
            linewidth=1,
        )
        
        # Add error bars
        ax.errorbar(
            metrics.total_tokens_per_episode,
            metrics.total_welfare_mean,
            yerr=metrics.total_welfare_std,
            fmt="none",
            ecolor=color,
            capsize=5,
            alpha=0.7,
        )
    
    ax.set_xlabel("Tokens per Episode (prompt + completion)", fontsize=12)
    ax.set_ylabel("Total Welfare", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title="Memory Type", loc="best")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cooperation_metrics(
    all_metrics: dict[str, ExperimentMetrics],
    output_path: Path,
    title: str = "Cooperation Metrics by Memory Condition",
) -> None:
    """Grouped bar plot of cooperation rate and stability."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mem_types = list(all_metrics.keys())
    x = np.arange(len(mem_types))
    width = 0.35
    
    rates = [m.cooperation_rate for m in all_metrics.values()]
    stability = [m.cooperation_stability for m in all_metrics.values()]
    
    bars1 = ax.bar(x - width/2, rates, width, label="Cooperation Rate", color=sns.color_palette("husl")[0])
    bars2 = ax.bar(x + width/2, stability, width, label="Stability", color=sns.color_palette("husl")[3])
    
    ax.set_xlabel("Memory Condition", fontsize=12)
    ax.set_ylabel("Score (0-1)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(mem_types, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_summary_table(
    all_metrics: dict[str, ExperimentMetrics],
    output_path: Path,
) -> None:
    """Create a summary table as an image."""
    
    fig, ax = plt.subplots(figsize=(12, len(all_metrics) * 0.8 + 2))
    ax.axis("off")
    
    columns = ["Memory Type", "Welfare", "Coop Rate", "Stability", "Gini", "Tokens/Ep"]
    rows = []
    
    for mem_type, metrics in all_metrics.items():
        rows.append([
            mem_type,
            f"{metrics.total_welfare_mean:.1f} ± {metrics.total_welfare_std:.1f}",
            f"{metrics.cooperation_rate:.2f}",
            f"{metrics.cooperation_stability:.2f}",
            f"{metrics.gini_mean:.3f}",
            f"{int(metrics.total_tokens_per_episode)}",
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colColours=["#f0f0f0"] * len(columns),
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Summary of Results by Memory Condition", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    results_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate all analysis plots and return paths."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_data = load_results(results_dir)
    
    if not all_data:
        print("No results found!")
        return []
    
    # Compute metrics for each condition
    all_metrics = {mem_type: compute_metrics(data) for mem_type, data in all_data.items()}
    
    generated_files = []
    
    # Generate plots
    plots = [
        ("cooperation_over_time.png", plot_cooperation_over_time),
        ("welfare_comparison.png", plot_welfare_comparison),
        ("inequality_comparison.png", plot_inequality_comparison),
        ("cost_vs_performance.png", plot_cost_vs_performance),
        ("cooperation_metrics.png", plot_cooperation_metrics),
    ]
    
    for filename, plot_func in plots:
        output_path = output_dir / filename
        plot_func(all_metrics, output_path)
        generated_files.append(output_path)
        print(f"  Generated {output_path}")
    
    # Generate summary table
    table_path = output_dir / "summary_table.png"
    create_summary_table(all_metrics, table_path)
    generated_files.append(table_path)
    print(f"  Generated {table_path}")
    
    return generated_files


if __name__ == "__main__":
    import click
    
    @click.command()
    @click.option("--results-dir", "-r", default="results", help="Results directory")
    @click.option("--output-dir", "-o", default="analysis/figures", help="Output directory")
    def main(results_dir: str, output_dir: str):
        """Generate analysis plots from experiment results."""
        generate_all_plots(Path(results_dir), Path(output_dir))
    
    main()

