"""
Run the Neutral Strategy Note Ablation Experiment.

Compares three conditions:
- None: Baseline (no memory)
- Hybrid: Trust table + cooperative seed ("Start by cooperating to establish trust.")
- Hybrid-Neutral: Trust table + neutral seed ("Decide each round based on the situation and observed outcomes.")

This experiment distinguishes between:
(a) The articulation effect (does having any strategy note help?)
(b) The priming effect (does the cooperative content matter?)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from environment.public_goods_env import PublicGoodsEnv, EnvConfig
from memory import create_memory
from memory.base_memory import RoundInfo
from agents.llm_policy import LLMPolicy, OpenAIBackend


def bootstrap_ci(data: list, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    bootstrapped = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    means = bootstrapped.mean(axis=1)
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return float(np.mean(data)), float(lower), float(upper)


def run_episode(env, agents, memories, memory_states, seed, episode_id):
    """Run a single episode and collect results."""
    observations = env.reset(seed=seed)
    
    contributions_log = []
    payoffs_log = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    done = False
    round_idx = 0
    
    while not done:
        actions = []
        for i, (agent, obs, mem_state) in enumerate(zip(agents, observations, memory_states)):
            action = agent.act(obs, mem_state, env.config.max_contribution)
            actions.append(action)
            total_prompt_tokens += action.prompt_tokens
            total_completion_tokens += action.completion_tokens
        
        contributions = [a.contribution for a in actions]
        result = env.step(contributions)
        observations = result.observations
        done = result.done
        
        contributions_log.append(contributions)
        payoffs_log.append(result.rewards)
        
        if not done:
            for i, (memory, obs, action) in enumerate(zip(memories, observations, actions)):
                round_info = RoundInfo(
                    round_index=round_idx,
                    agent_id=i,
                    contributions=contributions,
                    pot=result.info["pot"],
                    payoffs=result.rewards,
                    own_budget=obs.budget,
                    own_cumulative_payoff=obs.cumulative_payoff,
                )
                llm_response = action.raw_response if hasattr(action, "raw_response") else None
                memory_states[i] = memory.update(memory_states[i], round_info, llm_response)
        
        round_idx += 1
    
    all_contributions = [c for round_contribs in contributions_log for c in round_contribs]
    mean_contribution = sum(all_contributions) / len(all_contributions) if all_contributions else 0
    total_welfare = sum(sum(round_payoffs) for round_payoffs in payoffs_log)
    
    return {
        "episode_id": episode_id,
        "seed": seed,
        "contributions": contributions_log,
        "payoffs": payoffs_log,
        "total_welfare": total_welfare,
        "mean_contribution": mean_contribution,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
    }


def run_condition(memory_type: str, num_episodes: int = 10, base_seed: int = 300):
    """Run all episodes for a single memory condition."""
    print(f"\n=== Running {memory_type} condition ===")
    
    env_config = EnvConfig(
        num_agents=3,
        num_rounds=10,
        starting_budget=20.0,
        max_contribution=10,
        multiplier=1.8,
    )
    env = PublicGoodsEnv(env_config)
    backend = OpenAIBackend(model="gpt-4o-mini-2024-07-18", temperature=0.7)
    
    memory_kwargs = {
        "none": {},
        "hybrid": {"max_note_words": 20, "fair_share_fraction": 0.5},
        "hybrid_neutral": {"max_note_words": 20, "fair_share_fraction": 0.5},
    }.get(memory_type, {})
    
    results = []
    
    for ep in range(num_episodes):
        seed = base_seed + ep
        
        memories = []
        memory_states = []
        agents = []
        
        for agent_id in range(env_config.num_agents):
            memory = create_memory(memory_type, **memory_kwargs)
            memories.append(memory)
            memory_states.append(memory.init_state(agent_id, env_config.num_agents))
            
            agent = LLMPolicy(
                backend=backend,
                memory=memory,
                num_agents=env_config.num_agents,
                multiplier=env_config.multiplier,
            )
            agent.reset()
            agents.append(agent)
        
        try:
            episode_result = run_episode(
                env=env,
                agents=agents,
                memories=memories,
                memory_states=memory_states,
                seed=seed,
                episode_id=ep,
            )
            results.append(episode_result)
            print(f"  Episode {ep + 1}/{num_episodes}: welfare={episode_result['total_welfare']:.1f}, "
                  f"mean_contrib={episode_result['mean_contribution']:.1f}")
        except Exception as e:
            print(f"  Episode {ep + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def main():
    """Run the neutral note ablation experiment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file")
        sys.exit(1)
    
    print(f"API key found (starts with: {api_key[:8]}...)")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    conditions = ["none", "hybrid", "hybrid_neutral"]
    num_episodes = 10
    all_results = {}
    
    print("\n" + "=" * 60)
    print("RUNNING NEUTRAL NOTE ABLATION EXPERIMENT")
    print("=" * 60)
    
    for condition in conditions:
        results = run_condition(condition, num_episodes=num_episodes, base_seed=300)
        all_results[condition] = results
        
        data = {
            "config": {
                "num_agents": 3,
                "num_rounds": 10,
                "starting_budget": 20.0,
                "max_contribution": 10,
                "multiplier": 1.8,
                "memory_type": condition,
                "num_episodes": len(results),
                "base_seed": 300,
                "model": "gpt-4o-mini-2024-07-18",
                "temperature": 0.7,
                "experiment_type": "neutral_note_ablation",
            },
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }
        
        output_file = results_dir / f"neutral_ablation_{condition}.json"
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved to {output_file}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE - SUMMARY")
    print("=" * 60)
    
    for condition in conditions:
        welfare_list = [r["total_welfare"] for r in all_results[condition]]
        mean, ci_low, ci_high = bootstrap_ci(welfare_list)
        print(f"  {condition:15s}: {mean:6.1f} [{ci_low:.1f}, {ci_high:.1f}]")
    
    summary = {
        "experiment": "neutral_note_ablation",
        "timestamp": datetime.now().isoformat(),
        "conditions": {},
    }
    
    for condition in conditions:
        welfare_list = [r["total_welfare"] for r in all_results[condition]]
        contrib_list = [r["mean_contribution"] for r in all_results[condition]]
        mean_w, ci_low_w, ci_high_w = bootstrap_ci(welfare_list)
        mean_c, ci_low_c, ci_high_c = bootstrap_ci(contrib_list)
        summary["conditions"][condition] = {
            "welfare_mean": mean_w,
            "welfare_ci_low": ci_low_w,
            "welfare_ci_high": ci_high_w,
            "contribution_mean": mean_c,
            "contribution_ci_low": ci_low_c,
            "contribution_ci_high": ci_high_c,
            "num_episodes": len(welfare_list),
        }
    
    summary_file = results_dir / "neutral_ablation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
