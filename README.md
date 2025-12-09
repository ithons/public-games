# LLM Agent Memory in Repeated Social Dilemmas

**MIT 6.7960 Deep Learning — Fall 2025 Final Project**

## Research Question

*How much memory do LLM agents really need to maintain cooperation in repeated social dilemmas?*

This project investigates how different memory representations affect cooperation, welfare, and computational cost when LLM-based agents play a repeated public goods game.

## Key Finding

**Hybrid memory (trust table + strategy note) achieves 2× the welfare of all other conditions** by enabling perfect cooperation. While GPT-4o-mini agents maintain moderate cooperation (~50%) without memory, only the hybrid representation unlocks full cooperation.

| Memory Type | Welfare (95% CI) | Mean Contrib | Tokens/Episode |
|-------------|------------------|--------------|----------------|
| None | 120.3 [120.0, 120.7] | 5.01 | ~11,200 |
| Full History (k=5) | 120.5 [120.0, 121.4] | 5.02 | ~16,900 |
| Summary (50w) | 124.0 [122.1, 126.2] | 5.17 | ~13,600 |
| Structured | 123.0 [120.7, 126.0] | 5.13 | ~14,700 |
| **Hybrid** | **240.0 [240.0, 240.0]** | **10.00** | ~15,500 |

## Key Insights

1. **Hybrid achieves theoretical maximum welfare** (240) through perfect cooperation
2. **Full history provides NO benefit** over no memory (identical welfare)
3. **The strategy note is the critical component** — trust table alone doesn't help
4. **Effect is robust** across different multiplier values (α = 1.5, 1.8, 2.1)

## Project Structure

```
llm-games/
├── environment/            # Public goods game environment
│   └── public_goods_env.py
├── agents/                 # Agent policies
│   ├── base_policy.py
│   └── llm_policy.py       # LLM-based policy with OpenAI backend
├── memory/                 # Memory modules
│   ├── base_memory.py      # Abstract interface
│   ├── no_memory.py        # Baseline: no history
│   ├── full_history.py     # Last k rounds verbatim
│   ├── summary_memory.py   # LLM-generated rolling summary
│   ├── structured_memory.py # Trust table (numerical)
│   └── hybrid_memory.py    # Trust table + strategy note
├── experiments/            # Experiment runners
│   └── run_experiments.py
├── analysis/               # Analysis and plotting
│   ├── analyze_results.py  # Compute metrics with bootstrap CIs
│   └── figures/            # Generated plots
├── blog/                   # Technical blog (offline-safe)
│   ├── index.html
│   ├── style.css
│   └── assets/             # Figures for blog
├── results/                # Experiment results (JSON)
├── tests/                  # Unit tests
│   ├── test_environment.py
│   ├── test_memory.py
│   ├── test_sanity.py      # Payoff formula and metric verification
│   └── test_blog_consistency.py  # Verify blog matches data
├── tools/                  # Utility scripts
│   └── check_blog_consistency.py  # Automated blog verification
├── run_real_experiments.py # Main experiment script
└── requirements.txt
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Running Experiments

```bash
# Run all experiments (main + ablation + alpha sweep)
python run_real_experiments.py

# Or run individual conditions via CLI
python -m experiments.run_experiments --memory-type structured --episodes 15 --use-real-llm
python -m experiments.run_experiments --all-conditions --episodes 15 --use-real-llm
```

## Generating Analysis

```bash
# Generate plots and metrics summary
python -m analysis.analyze_results --results-dir results --output-dir analysis/figures

# Copy figures to blog
cp analysis/figures/*.png blog/assets/
```

## Running Tests

```bash
# Run all tests (includes blog consistency check)
python -m pytest tests/ -v

# Run sanity tests only
python -m pytest tests/test_sanity.py -v

# Run blog consistency checker directly
python tools/check_blog_consistency.py
```

## Cost Estimate

Running the full experiment suite (main + ablation + alpha sweep) with GPT-4o-mini:

| Experiment | Episodes | Tokens/Episode | Total Tokens | Est. Cost |
|------------|----------|----------------|--------------|-----------|
| Main (5 conditions × 15 ep) | 75 | ~13,500 avg | ~1M | ~$0.15 |
| Ablation (3 cond × 10 ep) | 30 | ~13,500 avg | ~400k | ~$0.06 |
| Alpha Sweep (3 α × 3 cond × 10 ep) | 90 | ~13,500 avg | ~1.2M | ~$0.18 |
| **Total** | 195 | — | ~2.6M | **~$0.40** |

For budget-constrained reproduction, run fewer episodes (e.g., 5 per condition) or test with mock agents first.

## Experimental Parameters

| Parameter | Value |
|-----------|-------|
| Model | GPT-4o-mini |
| Temperature | 0.7 |
| Agents (N) | 3 |
| Rounds (T) | 10 |
| Starting Budget (B) | 20.0 |
| Max Contribution (c_max) | 10 |
| Multiplier (α) | 1.8 (main), 1.5/2.1 (sweep) |
| Episodes | 15 (main), 10 (ablation/sweep) |

## Metrics

All metrics computed with 95% bootstrap confidence intervals (10,000 resamples):

- **Welfare**: Sum of payoffs across all agents and rounds per episode
- **Mean Contribution**: Average contribution per (agent, round)
- **Gini Coefficient**: Inequality of final budgets (0 = equal)
- **Token Cost**: Total prompt + completion tokens per episode

## Blog

Open `blog/index.html` in a browser to view the technical write-up. All assets are local — **renders fully offline**.

## Results Files

After running experiments, the `results/` directory contains:

- `none.json`, `full_history.json`, `summary.json`, `structured.json`, `hybrid.json` — Main experiments
- `ablation_*.json` — Ablation comparing none vs structured vs hybrid
- `sweep_alpha*_*.json` — Alpha sweep for robustness testing
- `summary_metrics.json` — Computed metrics for all conditions

## License

MIT License — for educational purposes.
