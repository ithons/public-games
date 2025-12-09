# LLM Agent Memory in Repeated Social Dilemmas

**MIT 6.7960 Deep Learning — Fall 2025 Final Project**

## Research Question

*How much memory do LLM agents really need to maintain cooperation in repeated social dilemmas?*

This project investigates how different memory representations affect cooperation, welfare, and computational cost when LLM-based agents play a repeated public-goods game.

## Key Finding

**Hybrid memory (trust table + strategy note) achieves 2x the welfare of other conditions** by enabling full cooperation. While GPT-4o-mini agents naturally maintain moderate cooperation (~50%) without memory, the hybrid representation unlocks emergent coordination.

| Memory Type | Welfare | Tokens/Episode |
|-------------|---------|----------------|
| None | 120.2 ± 0.6 | ~11,200 |
| Full History (k=5) | 121.8 ± 3.9 | ~16,900 |
| Summary (50w) | 122.9 ± 3.8 | ~13,600 |
| Structured | 125.4 ± 6.9 | ~14,700 |
| **Hybrid** | **240.0 ± 0.0** | ~15,500 |

## Hypotheses & Results

1. **Structure Helps**: ✅ **Strongly Supported** — Hybrid memory (structured + strategy note) dramatically outperforms other conditions.

2. **Diminishing Returns**: ✅ **Partially Supported** — Full history uses 51% more tokens than baseline for marginal improvement.

3. **Summaries in the Middle**: ✅ **Supported** — Summary memory occupies the expected intermediate position.

## Project Structure

```
llm-games/
├── environment/            # Public goods environment
│   └── public_goods_env.py
├── agents/                 # Agent policies
│   ├── base_policy.py
│   └── llm_policy.py
├── memory/                 # Memory modules
│   ├── base_memory.py
│   ├── no_memory.py
│   ├── full_history.py
│   ├── summary_memory.py
│   ├── structured_memory.py
│   └── hybrid_memory.py
├── experiments/            # Experiment runners
│   ├── run_experiments.py
│   └── configs/
├── analysis/               # Analysis and plotting
│   ├── analyze_results.py
│   └── figures/
├── blog/                   # Technical blog
│   ├── index.html
│   └── assets/
└── tests/                  # Unit tests
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

```bash
# Run with a specific memory condition
python -m experiments.run_experiments --memory-type structured --episodes 50 --seed 42

# Run all conditions
python -m experiments.run_experiments --all-conditions --episodes 50
```

## Generating Analysis

```bash
python -m analysis.analyze_results --results-dir results/ --output-dir analysis/figures/
```

## Blog

Open `blog/index.html` in a browser to view the technical write-up. All assets are local — no network required.

## License

MIT License — for educational purposes.

