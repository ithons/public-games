"""
Analysis and Plotting Utilities

Load experiment results, compute metrics with bootstrap confidence intervals, 
and generate figures for the blog.

Metric Definitions:
- Cooperation Rate: Fraction of (agent, round) events where contribution >= 0.5 * c_max
- Stability: 1 - Var(μ_t) / Var_max, where Var_max is variance of alternating [0, c_max, ...]
- Welfare: Sum of all payoffs across agents and rounds per episode
- Gini: Gini coefficient of final budgets (0 = perfect equality, 1 = max inequality)
- Tokens: Total prompt + completion tokens per episode
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

# Color palette
COLORS = {
    "none": "#f77189",
    "full_history": "#50b131", 
    "summary": "#3ba3ec",
    "structured": "#f9a630",
    "hybrid": "#a48cf4",
}
COLOR_LIST = ["#f77189", "#50b131", "#3ba3ec", "#f9a630", "#a48cf4", "#36ada4"]

# Set style for publication-quality figures
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


@dataclass
class MetricWithCI:
    """A metric value with 95% confidence interval via bootstrap."""
    mean: float
    ci_low: float
    ci_high: float
    std: float
    
    def __str__(self):
        return f"{self.mean:.2f} (95% CI [{self.ci_low:.2f}, {self.ci_high:.2f}])"
    
    def to_dict(self):
        return {
            "mean": round(self.mean, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "std": round(self.std, 4),
        }


@dataclass
class ExperimentMetrics:
    """Computed metrics for a single experiment condition."""
    memory_type: str
    num_episodes: int
    num_rounds: int
    num_agents: int
    c_max: int
    
    # Per-round metrics (averaged across episodes)
    mean_contribution_per_round: list[float] = field(default_factory=list)
    ci_contribution_per_round: list[tuple[float, float]] = field(default_factory=list)
    
    # Main metrics with confidence intervals
    welfare: Optional[MetricWithCI] = None
    cooperation_rate: Optional[MetricWithCI] = None
    stability: Optional[MetricWithCI] = None
    gini: Optional[MetricWithCI] = None
    tokens: Optional[MetricWithCI] = None
    mean_contribution: Optional[MetricWithCI] = None


def bootstrap_ci(values: list[float], n_bootstrap: int = 10000, ci: float = 0.95) -> MetricWithCI:
    """Compute mean and 95% confidence interval via bootstrap resampling."""
    if not values or len(values) == 0:
        return MetricWithCI(mean=0.0, ci_low=0.0, ci_high=0.0, std=0.0)
    
    values = np.array(values)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    
    if len(values) < 2:
        return MetricWithCI(mean=mean_val, ci_low=mean_val, ci_high=mean_val, std=std_val)
    
    # Bootstrap
    rng = np.random.default_rng(42)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(bootstrap_means, alpha * 100))
    ci_high = float(np.percentile(bootstrap_means, (1 - alpha) * 100))
    
    return MetricWithCI(mean=mean_val, ci_low=ci_low, ci_high=ci_high, std=std_val)


def gini_coefficient(values: list[float]) -> float:
    """
    Compute Gini coefficient for inequality measurement.
    
    Gini = 0 means perfect equality (all have the same)
    Gini approaching 1 means maximum inequality (one has all)
    """
    if not values or len(values) < 2:
        return 0.0
    
    sorted_values = np.array(sorted(values), dtype=float)
    n = len(sorted_values)
    
    # Handle case where all values are zero or equal
    if sorted_values.sum() == 0:
        return 0.0
    
    # Standard Gini formula
    cumsum = np.cumsum(sorted_values)
    return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n)


def compute_stability(mean_per_round: list[float], c_max: int) -> float:
    """
    Compute stability as 1 - Var(μ_t) / Var_max.
    
    Var_max is the variance of an ideal maximally unstable sequence:
    alternating [0, c_max, 0, c_max, ...], which has variance = (c_max/2)^2 = c_max^2/4
    
    Returns a value in [0, 1] where 1 = perfectly stable, 0 = maximally unstable.
    """
    if not mean_per_round or len(mean_per_round) < 2:
        return 1.0
    
    var_actual = float(np.var(mean_per_round))
    var_max = (c_max / 2) ** 2  # Variance of [0, c_max, 0, c_max, ...]
    
    if var_max == 0:
        return 1.0
    
    stability = 1 - var_actual / var_max
    return max(0.0, min(1.0, stability))


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load all experiment results from a directory."""
    results = {}
    
    # Only load main condition files (not ablation or sweep files)
    main_conditions = ["none", "full_history", "summary", "structured", "hybrid"]
    
    for json_file in results_dir.glob("*.json"):
        # Skip meta files and sweep files
        if json_file.stem in main_conditions:
            with open(json_file) as f:
                data = json.load(f)
            memory_type = data["config"]["memory_type"]
            results[memory_type] = data
    
    return results


def load_ablation_results(results_dir: Path) -> dict[str, dict]:
    """Load ablation experiment results."""
    results = {}
    for json_file in results_dir.glob("ablation_*.json"):
        with open(json_file) as f:
            data = json.load(f)
        # Extract condition name from filename
        condition = json_file.stem.replace("ablation_", "")
        results[condition] = data
    return results


def load_sweep_results(results_dir: Path, sweep_type: str = "alpha") -> dict[str, dict]:
    """Load sweep experiment results."""
    results = {}
    for json_file in results_dir.glob(f"sweep_{sweep_type}*.json"):
        with open(json_file) as f:
            data = json.load(f)
        # Extract key from filename (e.g., "sweep_alpha1.8_hybrid" -> "alpha1.8_hybrid")
        key = json_file.stem.replace("sweep_", "")
        results[key] = data
    return results


def compute_metrics(data: dict) -> ExperimentMetrics:
    """Compute aggregate metrics from experiment results with bootstrap CIs."""
    config = data["config"]
    results = data["results"]
    
    num_episodes = len(results)
    num_rounds = config["num_rounds"]
    num_agents = config["num_agents"]
    c_max = config["max_contribution"]
    
    metrics = ExperimentMetrics(
        memory_type=config["memory_type"],
        num_episodes=num_episodes,
        num_rounds=num_rounds,
        num_agents=num_agents,
        c_max=c_max,
    )
    
    # Collect per-round contributions across episodes
    # Structure: round_contributions[round_idx] = list of (agent contributions across all episodes)
    round_contributions = [[] for _ in range(num_rounds)]
    
    welfare_values = []
    gini_values = []
    coop_rate_values = []
    stability_values = []
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    mean_contrib_values = []
    
    for episode in results:
        contributions = episode["contributions"]
        
        # Per-round mean contributions for this episode
        episode_round_means = []
        episode_all_contribs = []
        
        for r, round_contribs in enumerate(contributions):
            if r < num_rounds:
                round_contributions[r].extend(round_contribs)
                episode_round_means.append(np.mean(round_contribs))
                episode_all_contribs.extend(round_contribs)
        
        # Welfare
        welfare_values.append(episode["total_welfare"])
        
        # Mean contribution for this episode
        if episode_all_contribs:
            mean_contrib_values.append(np.mean(episode_all_contribs))
        
        # Cooperation rate for this episode (fraction >= 0.5 * c_max)
        high_threshold = c_max * 0.5
        if episode_all_contribs:
            coop_rate = sum(1 for c in episode_all_contribs if c >= high_threshold) / len(episode_all_contribs)
            coop_rate_values.append(coop_rate)
        
        # Stability for this episode
        if episode_round_means:
            stab = compute_stability(episode_round_means, c_max)
            stability_values.append(stab)
        
        # Gini for this episode
        final_budgets = episode.get("final_budgets", [])
        if final_budgets:
            gini_values.append(gini_coefficient(final_budgets))
        
        # Token counts
        pt = episode.get("total_prompt_tokens", 0)
        ct = episode.get("total_completion_tokens", 0)
        prompt_tokens.append(pt)
        completion_tokens.append(ct)
        total_tokens.append(pt + ct)
    
    # Compute per-round statistics with CIs
    for r, contribs in enumerate(round_contributions):
        if contribs:
            ci = bootstrap_ci(contribs, n_bootstrap=5000)
            metrics.mean_contribution_per_round.append(ci.mean)
            metrics.ci_contribution_per_round.append((ci.ci_low, ci.ci_high))
        else:
            metrics.mean_contribution_per_round.append(0.0)
            metrics.ci_contribution_per_round.append((0.0, 0.0))
    
    # Compute main metrics with bootstrap CIs
    metrics.welfare = bootstrap_ci(welfare_values)
    metrics.cooperation_rate = bootstrap_ci(coop_rate_values)
    metrics.stability = bootstrap_ci(stability_values)
    metrics.gini = bootstrap_ci(gini_values)
    metrics.tokens = bootstrap_ci(total_tokens)
    metrics.mean_contribution = bootstrap_ci(mean_contrib_values)
    
    return metrics


def plot_cooperation_over_time(
    all_metrics: dict[str, ExperimentMetrics],
    output_path: Path,
    title: str = "Cooperation Over Time by Memory Condition",
) -> None:
    """Plot average contribution per round for each memory condition."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for mem_type, metrics in all_metrics.items():
        color = COLORS.get(mem_type, COLOR_LIST[0])
        rounds = list(range(1, len(metrics.mean_contribution_per_round) + 1))
        means = metrics.mean_contribution_per_round
        ci_low = [ci[0] for ci in metrics.ci_contribution_per_round]
        ci_high = [ci[1] for ci in metrics.ci_contribution_per_round]
        
        ax.plot(rounds, means, label=mem_type, color=color, linewidth=2, marker="o", markersize=4)
        ax.fill_between(rounds, ci_low, ci_high, alpha=0.2, color=color)
    
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean Contribution")
    ax.set_title(title)
    ax.legend(title="Memory Type", loc="best")
    if all_metrics:
        first_metrics = next(iter(all_metrics.values()))
        ax.set_xticks(range(1, first_metrics.num_rounds + 1))
    ax.set_ylim(0, None)
    
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
    means = [m.welfare.mean for m in all_metrics.values()]
    ci_low = [m.welfare.ci_low for m in all_metrics.values()]
    ci_high = [m.welfare.ci_high for m in all_metrics.values()]
    errors = [[m - l for m, l in zip(means, ci_low)], 
              [h - m for m, h in zip(means, ci_high)]]
    
    colors = [COLORS.get(mt, COLOR_LIST[0]) for mt in mem_types]
    bars = ax.bar(mem_types, means, yerr=errors, capsize=5, color=colors, edgecolor="black", linewidth=1)
    
    ax.set_xlabel("Memory Condition")
    ax.set_ylabel("Total Welfare (sum of payoffs)")
    ax.set_title(title)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.annotate(
            f"{mean:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=10,
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
    means = [m.gini.mean for m in all_metrics.values()]
    ci_low = [m.gini.ci_low for m in all_metrics.values()]
    ci_high = [m.gini.ci_high for m in all_metrics.values()]
    errors = [[m - l for m, l in zip(means, ci_low)], 
              [h - m for m, h in zip(means, ci_high)]]
    
    colors = [COLORS.get(mt, COLOR_LIST[0]) for mt in mem_types]
    bars = ax.bar(mem_types, means, yerr=errors, capsize=5, color=colors, edgecolor="black", linewidth=1)
    
    ax.set_xlabel("Memory Condition")
    ax.set_ylabel("Gini Coefficient (0=equal, 1=unequal)")
    ax.set_title(title)
    ax.set_ylim(0, max(0.1, max(means) * 1.5 if means else 0.1))
    
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
    
    for mem_type, metrics in all_metrics.items():
        color = COLORS.get(mem_type, COLOR_LIST[0])
        
        ax.scatter(
            metrics.tokens.mean,
            metrics.welfare.mean,
            s=150, c=[color], label=mem_type,
            edgecolors="black", linewidth=1,
        )
        
        # Add error bars for both dimensions
        ax.errorbar(
            metrics.tokens.mean, metrics.welfare.mean,
            xerr=[[metrics.tokens.mean - metrics.tokens.ci_low], 
                  [metrics.tokens.ci_high - metrics.tokens.mean]],
            yerr=[[metrics.welfare.mean - metrics.welfare.ci_low], 
                  [metrics.welfare.ci_high - metrics.welfare.mean]],
            fmt="none", ecolor=color, capsize=5, alpha=0.7,
        )
    
    ax.set_xlabel("Tokens per Episode (prompt + completion)")
    ax.set_ylabel("Total Welfare")
    ax.set_title(title)
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
    
    rates = [m.cooperation_rate.mean for m in all_metrics.values()]
    rate_errors = [
        [m.cooperation_rate.mean - m.cooperation_rate.ci_low for m in all_metrics.values()],
        [m.cooperation_rate.ci_high - m.cooperation_rate.mean for m in all_metrics.values()],
    ]
    
    stability = [m.stability.mean for m in all_metrics.values()]
    stab_errors = [
        [m.stability.mean - m.stability.ci_low for m in all_metrics.values()],
        [m.stability.ci_high - m.stability.mean for m in all_metrics.values()],
    ]
    
    bars1 = ax.bar(x - width/2, rates, width, yerr=rate_errors, capsize=3,
                   label="Cooperation Rate", color=COLOR_LIST[0])
    bars2 = ax.bar(x + width/2, stability, width, yerr=stab_errors, capsize=3,
                   label="Stability", color=COLOR_LIST[3])
    
    ax.set_xlabel("Memory Condition")
    ax.set_ylabel("Score (0-1)")
    ax.set_title(title)
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
    
    fig, ax = plt.subplots(figsize=(14, len(all_metrics) * 0.8 + 2))
    ax.axis("off")
    
    columns = ["Memory Type", "Welfare", "Mean Contrib", "Coop Rate", "Stability", "Gini", "Tokens/Ep"]
    rows = []
    
    for mem_type, metrics in all_metrics.items():
        rows.append([
            mem_type,
            f"{metrics.welfare.mean:.1f} [{metrics.welfare.ci_low:.1f}, {metrics.welfare.ci_high:.1f}]",
            f"{metrics.mean_contribution.mean:.2f}",
            f"{metrics.cooperation_rate.mean:.2f}",
            f"{metrics.stability.mean:.2f}",
            f"{metrics.gini.mean:.3f}",
            f"{int(metrics.tokens.mean):,}",
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colColours=["#f0f0f0"] * len(columns),
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.title("Summary of Results by Memory Condition\n(values show mean with 95% CI)", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def export_metrics_json(
    all_metrics: dict[str, ExperimentMetrics],
    output_path: Path,
) -> None:
    """Export metrics to JSON for use in blog updates."""
    
    export_data = {}
    
    for mem_type, metrics in all_metrics.items():
        export_data[mem_type] = {
            "condition": mem_type,
            "num_episodes": metrics.num_episodes,
            "welfare": metrics.welfare.to_dict(),
            "mean_contribution": metrics.mean_contribution.to_dict(),
            "cooperation_rate": metrics.cooperation_rate.to_dict(),
            "stability": metrics.stability.to_dict(),
            "gini": metrics.gini.to_dict(),
            "tokens": metrics.tokens.to_dict(),
        }
    
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)
    
    print(f"  Exported metrics to {output_path}")


def plot_ablation_comparison(
    ablation_metrics: dict[str, ExperimentMetrics],
    output_path: Path,
    title: str = "Hybrid vs Structured Ablation",
) -> None:
    """Bar plot comparing ablation conditions."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = list(ablation_metrics.keys())
    means = [m.welfare.mean for m in ablation_metrics.values()]
    errors = [
        [m.welfare.mean - m.welfare.ci_low for m in ablation_metrics.values()],
        [m.welfare.ci_high - m.welfare.mean for m in ablation_metrics.values()],
    ]
    
    colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(conditions))]
    bars = ax.bar(conditions, means, yerr=errors, capsize=5, color=colors, edgecolor="black", linewidth=1)
    
    ax.set_xlabel("Condition")
    ax.set_ylabel("Total Welfare")
    ax.set_title(title)
    
    for bar, mean in zip(bars, means):
        ax.annotate(f"{mean:.1f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10)
    
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_alpha_sweep(
    sweep_results: dict[str, dict],
    output_path: Path,
    title: str = "Welfare vs Multiplier (α) by Memory Condition",
) -> None:
    """Line plot showing welfare vs alpha for different memory conditions."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Parse results by condition and alpha
    by_condition = {}
    for key, data in sweep_results.items():
        # Parse key like "alpha1.8_hybrid"
        parts = key.split("_", 1)
        if len(parts) == 2:
            alpha_str, condition = parts
            alpha = float(alpha_str.replace("alpha", ""))
            
            if condition not in by_condition:
                by_condition[condition] = {"alphas": [], "welfare_means": [], "welfare_errors": []}
            
            metrics = compute_metrics(data)
            by_condition[condition]["alphas"].append(alpha)
            by_condition[condition]["welfare_means"].append(metrics.welfare.mean)
            by_condition[condition]["welfare_errors"].append(
                (metrics.welfare.mean - metrics.welfare.ci_low, 
                 metrics.welfare.ci_high - metrics.welfare.mean)
            )
    
    for condition, cond_data in by_condition.items():
        # Sort by alpha
        sorted_idx = np.argsort(cond_data["alphas"])
        alphas = [cond_data["alphas"][i] for i in sorted_idx]
        means = [cond_data["welfare_means"][i] for i in sorted_idx]
        errors_low = [cond_data["welfare_errors"][i][0] for i in sorted_idx]
        errors_high = [cond_data["welfare_errors"][i][1] for i in sorted_idx]
        
        color = COLORS.get(condition, COLOR_LIST[0])
        ax.errorbar(alphas, means, yerr=[errors_low, errors_high], 
                    label=condition, color=color, marker="o", capsize=5, linewidth=2)
    
    ax.set_xlabel("Multiplier (α)")
    ax.set_ylabel("Total Welfare")
    ax.set_title(title)
    ax.legend(title="Memory Type", loc="best")
    
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
    
    print(f"Loaded results for {len(all_data)} conditions: {list(all_data.keys())}")
    
    # Compute metrics for each condition
    all_metrics = {}
    for mem_type, data in all_data.items():
        metrics = compute_metrics(data)
        all_metrics[mem_type] = metrics
        print(f"  {mem_type}: welfare={metrics.welfare}, coop_rate={metrics.cooperation_rate}")
    
    generated_files = []
    
    # Generate main plots
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
    
    # Export metrics JSON
    metrics_path = results_dir / "summary_metrics.json"
    export_metrics_json(all_metrics, metrics_path)
    
    # Try to generate ablation plot if data exists
    ablation_data = load_ablation_results(results_dir)
    if ablation_data:
        ablation_metrics = {k: compute_metrics(v) for k, v in ablation_data.items()}
        ablation_path = output_dir / "ablation_hybrid_vs_structured.png"
        plot_ablation_comparison(ablation_metrics, ablation_path)
        generated_files.append(ablation_path)
        print(f"  Generated {ablation_path}")
    
    # Try to generate alpha sweep plot if data exists
    sweep_data = load_sweep_results(results_dir, "alpha")
    if sweep_data:
        sweep_path = output_dir / "alpha_sweep.png"
        plot_alpha_sweep(sweep_data, sweep_path)
        generated_files.append(sweep_path)
        print(f"  Generated {sweep_path}")
    
    return generated_files


if __name__ == "__main__":
    import click
    
    @click.command()
    @click.option("--results-dir", "-r", default="results", help="Results directory")
    @click.option("--output-dir", "-o", default="analysis/figures", help="Output directory")
    def main(results_dir: str, output_dir: str):
        """Generate analysis plots from experiment results."""
        print(f"Analyzing results from {results_dir}...")
        generate_all_plots(Path(results_dir), Path(output_dir))
        print("Done!")
    
    main()
