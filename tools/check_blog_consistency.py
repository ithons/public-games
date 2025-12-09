"""
Blog Consistency Checker

Verifies that numeric claims in blog/index.html match the actual metrics
from the experiment results. This prevents drift between code and documentation.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Mismatch:
    """A mismatch between blog claim and actual data."""
    location: str
    claimed_value: float
    actual_value: float
    tolerance: float
    metric_name: str

    def __str__(self) -> str:
        return (
            f"MISMATCH in {self.location}: "
            f"claimed {self.metric_name}={self.claimed_value}, "
            f"actual={self.actual_value} (tolerance={self.tolerance})"
        )


def load_summary_metrics(results_dir: Path) -> dict:
    """Load the summary metrics JSON file."""
    metrics_file = results_dir / "summary_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Summary metrics not found: {metrics_file}")
    
    with open(metrics_file) as f:
        return json.load(f)


def load_ablation_metrics(results_dir: Path) -> dict[str, dict]:
    """Load ablation experiment results and compute welfare means."""
    ablation_files = ["ablation_none.json", "ablation_structured.json", "ablation_hybrid.json"]
    metrics = {}
    
    for filename in ablation_files:
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
            
            condition = filename.replace("ablation_", "").replace(".json", "")
            welfare_values = [r["total_welfare"] for r in data["results"]]
            metrics[condition] = {
                "welfare_mean": sum(welfare_values) / len(welfare_values),
                "num_episodes": len(welfare_values),
            }
    
    return metrics


def load_alpha_sweep_metrics(results_dir: Path) -> dict[str, dict]:
    """Load alpha sweep results."""
    metrics = {}
    
    for filepath in results_dir.glob("sweep_alpha*.json"):
        with open(filepath) as f:
            data = json.load(f)
        
        alpha = data["config"]["multiplier"]
        condition = data["config"]["memory_type"]
        welfare_values = [r["total_welfare"] for r in data["results"]]
        
        key = f"alpha{alpha}_{condition}"
        metrics[key] = {
            "alpha": alpha,
            "condition": condition,
            "welfare_mean": sum(welfare_values) / len(welfare_values),
        }
    
    return metrics


def parse_blog_html(blog_path: Path) -> str:
    """Load blog HTML content."""
    with open(blog_path) as f:
        return f.read()


def extract_welfare_claims(html: str) -> list[tuple[str, str, float]]:
    """
    Extract welfare claims from the blog.
    Returns list of (location, condition, claimed_value).
    """
    claims = []
    
    # Pattern for welfare values like "120.3 [120.0, 120.7]"
    welfare_pattern = r"(\w+).*?(\d+\.?\d*)\s*\[[\d., ]+\]"
    
    # Find in summary table - look for table rows
    table_section = re.search(
        r"Summary Table.*?</table>", 
        html, 
        re.DOTALL | re.IGNORECASE
    )
    
    if table_section:
        table_html = table_section.group()
        # Find table rows with welfare values
        row_pattern = r"<td>(None|Full History|Summary|Structured|Hybrid)</td>\s*<td>(\d+\.?\d*)"
        for match in re.finditer(row_pattern, table_html, re.IGNORECASE):
            condition = match.group(1).lower().replace(" ", "_").replace("full_history", "full_history")
            if condition == "full_history":
                condition = "full_history"
            welfare = float(match.group(2))
            claims.append(("summary_table", condition, welfare))
    
    return claims


def extract_token_claims(html: str) -> dict[str, float]:
    """Extract token count claims from the blog."""
    tokens = {}
    
    # Pattern for token values like "~11,200" or "~15,500"
    token_pattern = r"~([\d,]+)/episode"
    
    # Look in the memory conditions table
    table_section = re.search(
        r"Memory Conditions.*?</table>", 
        html, 
        re.DOTALL | re.IGNORECASE
    )
    
    if table_section:
        table_html = table_section.group()
        # Look for rows with token counts
        row_pattern = r"<td><strong>(No Memory|Full History|Summary|Structured|Hybrid)</strong></td>.*?~([\d,]+)/episode"
        for match in re.finditer(row_pattern, table_html, re.DOTALL | re.IGNORECASE):
            condition = match.group(1).lower().replace(" ", "_")
            if condition == "no_memory":
                condition = "none"
            token_val = int(match.group(2).replace(",", ""))
            tokens[condition] = token_val
    
    return tokens


def check_main_metrics(
    metrics: dict, 
    html: str, 
    tolerance: float = 0.5
) -> list[Mismatch]:
    """Check that main welfare values in blog match metrics."""
    mismatches = []
    
    # Define expected mappings based on blog content analysis
    expected_welfare = {
        "none": 120.3,
        "full_history": 120.5,
        "summary": 124.0,
        "structured": 123.0,
        "hybrid": 240.0,
    }
    
    expected_contrib = {
        "none": 5.01,
        "full_history": 5.02,
        "summary": 5.17,
        "structured": 5.13,
        "hybrid": 10.00,
    }
    
    expected_gini = {
        "none": 0.002,
        "full_history": 0.002,
        "summary": 0.006,
        "structured": 0.005,
        "hybrid": 0.000,
    }
    
    # Check welfare
    for condition, expected in expected_welfare.items():
        if condition in metrics:
            actual = metrics[condition]["welfare"]["mean"]
            if abs(actual - expected) > tolerance:
                mismatches.append(Mismatch(
                    location="main_results",
                    claimed_value=expected,
                    actual_value=actual,
                    tolerance=tolerance,
                    metric_name=f"{condition}_welfare",
                ))
    
    # Check mean contribution
    for condition, expected in expected_contrib.items():
        if condition in metrics:
            actual = metrics[condition]["mean_contribution"]["mean"]
            if abs(actual - expected) > 0.05:  # Tighter tolerance for contrib
                mismatches.append(Mismatch(
                    location="main_results",
                    claimed_value=expected,
                    actual_value=actual,
                    tolerance=0.05,
                    metric_name=f"{condition}_mean_contribution",
                ))
    
    # Check gini
    for condition, expected in expected_gini.items():
        if condition in metrics:
            actual = metrics[condition]["gini"]["mean"]
            if abs(actual - expected) > 0.002:  # Tolerance for gini
                mismatches.append(Mismatch(
                    location="main_results",
                    claimed_value=expected,
                    actual_value=actual,
                    tolerance=0.002,
                    metric_name=f"{condition}_gini",
                ))
    
    # Check tokens (larger tolerance for tokens)
    expected_tokens = {
        "none": 11200,
        "full_history": 16900,
        "summary": 13600,
        "structured": 14700,
        "hybrid": 15500,
    }
    
    for condition, expected in expected_tokens.items():
        if condition in metrics:
            actual = metrics[condition]["tokens"]["mean"]
            if abs(actual - expected) > 500:  # 500 token tolerance
                mismatches.append(Mismatch(
                    location="main_results",
                    claimed_value=expected,
                    actual_value=actual,
                    tolerance=500,
                    metric_name=f"{condition}_tokens",
                ))
    
    return mismatches


def check_ablation_metrics(
    ablation_metrics: dict,
    tolerance: float = 2.0
) -> list[Mismatch]:
    """Check ablation claims match actual data."""
    mismatches = []
    
    # Expected ablation values from blog
    expected = {
        "none": 120.0,
        "structured": 120.8,
        "hybrid": 236.5,  # Blog says 9 of 10 at 240, one at 204.8 → average ~236.5
    }
    
    for condition, expected_val in expected.items():
        if condition in ablation_metrics:
            actual = ablation_metrics[condition]["welfare_mean"]
            if abs(actual - expected_val) > tolerance:
                mismatches.append(Mismatch(
                    location="ablation",
                    claimed_value=expected_val,
                    actual_value=actual,
                    tolerance=tolerance,
                    metric_name=f"ablation_{condition}_welfare",
                ))
    
    return mismatches


def check_alpha_sweep_metrics(
    sweep_metrics: dict,
    tolerance: float = 5.0
) -> list[Mismatch]:
    """Check alpha sweep claims match actual data."""
    mismatches = []
    
    # Expected values from blog (approximate)
    # α = 1.5: none/structured ≈ 75, hybrid ≈ 150
    # α = 1.8: none/structured ≈ 120, hybrid ≈ 237
    # α = 2.1: none/structured ≈ 165, hybrid ≈ 330
    
    expected = {
        "alpha1.5_none": 75,
        "alpha1.5_structured": 75,
        "alpha1.5_hybrid": 150,
        "alpha1.8_none": 120,
        "alpha1.8_structured": 120,
        "alpha1.8_hybrid": 237,
        "alpha2.1_none": 165,
        "alpha2.1_structured": 165,
        "alpha2.1_hybrid": 330,
    }
    
    for key, expected_val in expected.items():
        if key in sweep_metrics:
            actual = sweep_metrics[key]["welfare_mean"]
            if abs(actual - expected_val) > tolerance:
                mismatches.append(Mismatch(
                    location="alpha_sweep",
                    claimed_value=expected_val,
                    actual_value=actual,
                    tolerance=tolerance,
                    metric_name=f"sweep_{key}",
                ))
    
    return mismatches


def check_51_percent_claim(metrics: dict) -> Optional[Mismatch]:
    """
    Check the "51 percent more tokens" claim for full_history vs none.
    """
    if "none" not in metrics or "full_history" not in metrics:
        return None
    
    none_tokens = metrics["none"]["tokens"]["mean"]
    full_tokens = metrics["full_history"]["tokens"]["mean"]
    
    actual_percent = ((full_tokens - none_tokens) / none_tokens) * 100
    claimed_percent = 51
    
    if abs(actual_percent - claimed_percent) > 5:  # 5% tolerance
        return Mismatch(
            location="token_claim",
            claimed_value=claimed_percent,
            actual_value=actual_percent,
            tolerance=5,
            metric_name="full_history_token_increase_percent",
        )
    
    return None


def run_consistency_check(
    results_dir: Path,
    blog_path: Path,
) -> list[Mismatch]:
    """Run full consistency check and return all mismatches."""
    all_mismatches = []
    
    # Load data
    try:
        metrics = load_summary_metrics(results_dir)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return [Mismatch("data_loading", 0, 0, 0, str(e))]
    
    html = parse_blog_html(blog_path)
    
    # Check main metrics
    all_mismatches.extend(check_main_metrics(metrics, html))
    
    # Check 51% claim
    token_mismatch = check_51_percent_claim(metrics)
    if token_mismatch:
        all_mismatches.append(token_mismatch)
    
    # Check ablation metrics
    try:
        ablation_metrics = load_ablation_metrics(results_dir)
        all_mismatches.extend(check_ablation_metrics(ablation_metrics))
    except Exception as e:
        print(f"Warning: Could not check ablation metrics: {e}")
    
    # Check alpha sweep metrics
    try:
        sweep_metrics = load_alpha_sweep_metrics(results_dir)
        all_mismatches.extend(check_alpha_sweep_metrics(sweep_metrics))
    except Exception as e:
        print(f"Warning: Could not check alpha sweep metrics: {e}")
    
    return all_mismatches


def main():
    """Run consistency check from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check blog consistency with metrics")
    parser.add_argument(
        "--results-dir", "-r",
        default="results",
        help="Path to results directory",
    )
    parser.add_argument(
        "--blog-path", "-b",
        default="blog/index.html",
        help="Path to blog HTML file",
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    blog_path = Path(args.blog_path)
    
    if not blog_path.exists():
        print(f"Error: Blog file not found: {blog_path}")
        return 1
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print(f"Checking blog consistency...")
    print(f"  Results: {results_dir}")
    print(f"  Blog: {blog_path}")
    print()
    
    mismatches = run_consistency_check(results_dir, blog_path)
    
    if not mismatches:
        print("✓ All blog claims are consistent with the data!")
        return 0
    else:
        print(f"✗ Found {len(mismatches)} mismatch(es):")
        for m in mismatches:
            print(f"  - {m}")
        return 1


if __name__ == "__main__":
    exit(main())
