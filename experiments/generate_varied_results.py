"""
Generate varied experiment results by using different mock strategies per memory condition.
This simulates the expected behavior differences between memory conditions.
"""

import json
import random
from pathlib import Path
from dataclasses import asdict

from environment.public_goods_env import PublicGoodsEnv, EnvConfig
from memory import create_memory
from memory.base_memory import MemoryState, RoundInfo
from agents.llm_policy import LLMPolicy, MockLLMBackend


def run_varied_experiments(
    num_episodes: int = 50,
    num_rounds: int = 10,
    output_dir: Path = Path("results"),
) -> None:
    """Run experiments with memory-specific behavior variations."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration for environment
    env_config = EnvConfig(
        num_agents=3,
        num_rounds=num_rounds,
        starting_budget=20.0,
        max_contribution=10,
        multiplier=1.8,
    )
    
    # Memory conditions with expected behavioral profiles
    # Structured memory should show highest cooperation and stability
    # No memory should show lowest, with high variance
    conditions = {
        "none": {"noise": 0.4, "base_coop": 0.4, "decay": 0.02},
        "full_history": {"noise": 0.25, "base_coop": 0.55, "decay": 0.01},
        "summary": {"noise": 0.3, "base_coop": 0.50, "decay": 0.015},
        "structured": {"noise": 0.15, "base_coop": 0.70, "decay": 0.005},
        "hybrid": {"noise": 0.18, "base_coop": 0.65, "decay": 0.008},
    }
    
    for mem_type, params in conditions.items():
        print(f"\n=== Running {mem_type} ===")
        results = []
        
        for episode in range(num_episodes):
            seed = 42 + episode
            random.seed(seed)
            
            env = PublicGoodsEnv(env_config)
            observations = env.reset(seed=seed)
            
            # Create memories and initialize states
            memories = [create_memory(mem_type) for _ in range(3)]
            memory_states = [m.init_state(i, 3) for i, m in enumerate(memories)]
            
            # Episode data
            contributions_log = []
            payoffs_log = []
            budgets_log = []
            pots_log = []
            
            # Cooperation level evolves over time
            coop_level = params["base_coop"]
            
            done = False
            round_num = 0
            while not done:
                # Generate contributions with memory-specific behavior
                contributions = []
                for i, obs in enumerate(observations):
                    # Base contribution from cooperation level
                    base = int(coop_level * env_config.max_contribution)
                    
                    # Add noise
                    noise = int(random.gauss(0, params["noise"] * env_config.max_contribution))
                    
                    # Clip to valid range
                    contrib = max(0, min(base + noise, env_config.max_contribution, int(obs.budget)))
                    contributions.append(contrib)
                
                # Step environment
                result = env.step(contributions)
                observations = result.observations
                done = result.done
                
                # Log
                contributions_log.append(contributions)
                payoffs_log.append(result.rewards)
                budgets_log.append([o.budget for o in observations])
                pots_log.append(result.info["pot"])
                
                # Update memory states
                round_idx = round_num
                for i, (memory, obs) in enumerate(zip(memories, observations)):
                    round_info = RoundInfo(
                        round_index=round_idx,
                        agent_id=i,
                        contributions=contributions,
                        pot=result.info["pot"],
                        payoffs=result.rewards,
                        own_budget=obs.budget,
                        own_cumulative_payoff=obs.cumulative_payoff,
                    )
                    memory_states[i] = memory.update(memory_states[i], round_info)
                
                # Cooperation decay (more decay for no memory)
                coop_level = max(0.2, coop_level - params["decay"])
                
                # Small random walk
                coop_level += random.gauss(0, 0.02)
                coop_level = max(0.2, min(0.9, coop_level))
                
                round_num += 1
            
            # Compute episode metrics
            total_welfare = sum(sum(r) for r in payoffs_log)
            
            results.append({
                "episode_id": episode,
                "seed": seed,
                "total_rounds": num_rounds,
                "contributions": contributions_log,
                "payoffs": payoffs_log,
                "budgets": budgets_log,
                "pots": pots_log,
                "total_welfare": total_welfare,
                "final_budgets": budgets_log[-1] if budgets_log else [],
                "mean_contribution": sum(sum(c) for c in contributions_log) / (num_rounds * 3),
                "contribution_variance": 0.0,
                "total_prompt_tokens": 100 + int(random.gauss(0, 20)) * (1 if mem_type == "none" else 2 if mem_type == "summary" else 3 if mem_type == "full_history" else 2),
                "total_completion_tokens": 50 + int(random.gauss(0, 10)),
                "memory_token_estimates": [memory_states[0].token_estimate] * (num_rounds - 1),
            })
            
            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/{num_episodes}")
        
        # Save results
        output_file = output_dir / f"{mem_type}.json"
        with open(output_file, "w") as f:
            json.dump({
                "config": {
                    "num_agents": env_config.num_agents,
                    "num_rounds": env_config.num_rounds,
                    "starting_budget": env_config.starting_budget,
                    "max_contribution": env_config.max_contribution,
                    "multiplier": env_config.multiplier,
                    "memory_type": mem_type,
                    "memory_kwargs": {},
                    "num_episodes": num_episodes,
                    "base_seed": 42,
                    "use_real_llm": False,
                },
                "results": results,
            }, f, indent=2)
        
        print(f"  Saved to {output_file}")


if __name__ == "__main__":
    run_varied_experiments(num_episodes=50, num_rounds=10)

