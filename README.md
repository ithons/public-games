# LLM Agent Memory in Repeated Social Dilemmas

How much memory does an LLM agent need to keep cooperating in a repeated public goods game, and does the shape of that memory matter more than the amount? Three GPT-4o-mini agents play 10 rounds with a starting budget of 20, a contribution cap of 10, and a multiplier of 1.8, and the only thing that changes across conditions is the block of text spliced into the prompt.

MIT 6.7960 (Deep Learning) final project, Fall 2025.

Five memory representations share one interface (`init_state`, `update`, `render_for_prompt`, `estimate_tokens`), so the comparison is an ablation on the prompt rather than five different agents. Each module also reports its own token estimate, which means what a memory costs is measured by the same object that produces it.

| Memory | Welfare (95% CI) | Mean contribution | Tokens/episode |
|---|---|---|---|
| None | 120.3 [120.0, 120.7] | 5.01 | ~11,200 |
| Full history (k=5) | 120.5 [120.0, 121.4] | 5.02 | ~16,900 |
| Summary (50w) | 124.0 [122.1, 126.2] | 5.17 | ~13,600 |
| Structured (trust table) | 123.0 [120.7, 126.0] | 5.13 | ~14,700 |
| Hybrid (table + strategy note) | 240.0 [240.0, 240.0] | 10.00 | ~15,500 |

15 episodes per condition, 95% bootstrap CIs over 10,000 resamples.

## What the 2x actually is

Hybrid agents contribute 10 out of 10 every round. Everyone else contributes almost exactly 5. At a multiplier of 1.8 that is 240 versus 120, so the doubling is "played the maximum" against "played half", not a graded behavioral improvement. The alpha sweep backs this up: 75 against 150 at α=1.5, 120 against 236 at α=1.8, 165 against 330 at α=2.1. Exactly 2x every time, which is what you get from a ceiling effect, not from a mechanism that scales with the payoff.

And the reason hybrid hits the ceiling is less interesting than it first looks. Its strategy note is seeded from round one with the literal string "Start by cooperating to establish trust." Rerunning it with a neutral note drops welfare to 124.2, which is the no-memory baseline. So the honest claim is not that a richer memory representation produces cooperation. It is that a cooperative instruction carried inside the memory string produces cooperation, and the trust table on its own does essentially nothing (120.8 against a 120.0 baseline).

What survives that deflation is still worth knowing: full history is the clearest negative result here. It costs 50% more tokens than no memory and buys exactly nothing, 120.5 against 120.3. More context is not more coordination.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m experiments.run_experiments --memory structured --episodes 2 --backend mock
python run_real_experiments.py                    # the full condition sweep, needs OPENAI_API_KEY
python run_neutral_note_experiment.py             # the neutral strategy-note ablation
python -m analysis.analyze_results --logs results/<file>.json
python -m pytest tests/ -q
```

The mock backend has cooperative, defector, random, and tit-for-tat presets, so the whole pipeline and the test suite run with no API key and no spend. Use it before spending anything: the real sweep is 5 conditions against gpt-4o-mini.

The environment refuses to construct a game that is not a dilemma. `PublicGoodsConfig` raises unless `1 < multiplier < num_agents`, which rules out the parameter settings where cooperating or defecting is trivially dominant. Contributions are clamped both in the response parser and again in `env.step`, so a hallucinated "contribution: 900" cannot corrupt payoff accounting.

## License

MIT
