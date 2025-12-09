# LLM Agent Memory in Repeated Social Dilemmas

**MIT 6.7960 Deep Learning — Fall 2025 Final Project**

## Research Question

*How much memory do LLM agents really need to maintain cooperation in repeated social dilemmas?*

This project investigates how different memory representations affect cooperation, welfare, and computational cost when LLM-based agents play a repeated public-goods game.

## Hypotheses

1. **Structure Helps**: Structured, low-entropy memory (e.g., trust tables) maintains cooperation more stably and cheaply than raw dialogue history.

2. **Diminishing Returns**: Beyond a small memory capacity, adding more history yields sharply diminishing returns in cooperation while increasing token costs.

3. **Summaries in the Middle**: Natural-language summaries sit between structured memory and full logs in both effectiveness and cost.

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

