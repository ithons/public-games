"""
Run real experiments with OpenAI API.

This script runs experiments with all memory conditions using real LLM calls.
It also includes ablation experiments and robustness sweeps.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from environment.public_goods_env import PublicGoodsEnv, EnvConfig
from memory import create_memory, MEMORY_REGISTRY
from memory.base_memory import RoundInfo
from agents.llm_policy import LLMPolicy, OpenAIBackend


def run_episode(env, agents, memories, memory_states, seed, episode_id):
    """Run a single episode and collect results."""
    observations = env.reset(seed=seed)
    
    contributions_log = []
    payoffs_log = []
    budgets_log = []
    pots_log = []
    memory_tokens_log = []
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    done = False
    round_idx = 0
    
    while not done:
        # Collect actions from all agents
        actions = []
        for i, (agent, obs, mem_state) in enumerate(zip(agents, observations, memory_states)):
            action = agent.act(obs, mem_state, env.config.max_contribution)
            actions.append(action)
            total_prompt_tokens += action.prompt_tokens
            total_completion_tokens += action.completion_tokens
        
        contributions = [a.contribution for a in actions]
        
        # Execute step
        result = env.step(contributions)
        observations = result.observations
        done = result.done
        
        # Log round data
        contributions_log.append(contributions)
        payoffs_log.append(result.rewards)
        budgets_log.append([obs.budget for obs in observations])
        pots_log.append(result.info["pot"])
        
        # Update memories
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
            
            memory_tokens_log.append(sum(ms.token_estimate for ms in memory_states))
        
        round_idx += 1
    
    # Compute aggregate metrics
    all_contributions = [c for round_contribs in contributions_log for c in round_contribs]
    mean_contribution = sum(all_contributions) / len(all_contributions) if all_contributions else 0
    variance = sum((c - mean_contribution) ** 2 for c in all_contributions) / len(all_contributions) if all_contributions else 0
    
    total_welfare = sum(sum(round_payoffs) for round_payoffs in payoffs_log)
    final_budgets = budgets_log[-1] if budgets_log else []
    
    return {
        "episode_id": episode_id,
        "seed": seed,
        "total_rounds": env.config.num_rounds,
        "contributions": contributions_log,
        "payoffs": payoffs_log,
        "budgets": budgets_log,
        "pots": pots_log,
        "total_welfare": total_welfare,
        "final_budgets": final_budgets,
        "mean_contribution": mean_contribution,
        "contribution_variance": variance,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "memory_token_estimates": memory_tokens_log,
    }


def run_condition(
    memory_type: str, 
    memory_kwargs: dict, 
    num_episodes: int = 15, 
    base_seed: int = 42,
    env_config: Optional[EnvConfig] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
):
    """Run all episodes for a single memory condition."""
    print(f"\n=== Running {memory_type} memory condition ===")
    
    # Create environment
    if env_config is None:
        env_config = EnvConfig(
            num_agents=3,
            num_rounds=10,
            starting_budget=20.0,
            max_contribution=10,
            multiplier=1.8,
        )
    env = PublicGoodsEnv(env_config)
    
    # Create shared backend
    backend = OpenAIBackend(model=model, temperature=temperature)
    
    results = []
    
    for ep in range(num_episodes):
        seed = base_seed + ep
        
        # Create fresh memories and agents for each episode
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
        
        # Run episode
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
                  f"mean_contrib={episode_result['mean_contribution']:.1f}, "
                  f"tokens={episode_result['total_prompt_tokens'] + episode_result['total_completion_tokens']}")
        except Exception as e:
            print(f"  Episode {ep + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def get_memory_kwargs(memory_type: str) -> dict:
    """Get default kwargs for each memory type."""
    defaults = {
        "none": {},
        "full_history": {"window_size": 5},
        "summary": {"max_words": 50},
        "structured": {"fair_share_fraction": 0.5},
        "hybrid": {"max_note_words": 20, "fair_share_fraction": 0.5},
        "hybrid_neutral": {"max_note_words": 20, "fair_share_fraction": 0.5},
    }
    return defaults.get(memory_type, {})


def save_results(results_dir: Path, filename: str, config: dict, results: list):
    """Save experiment results to JSON file."""
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f"{filename}.json"
    
    data = {
        "config": config,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved to {output_file}")
    return output_file


def run_main_experiments(results_dir: Path, num_episodes: int = 15):
    """Run main experiments for all memory conditions."""
    print("\n" + "="*60)
    print("RUNNING MAIN EXPERIMENTS")
    print("="*60)
    
    memory_types = ["none", "full_history", "summary", "structured", "hybrid"]
    
    for mem_type in memory_types:
        memory_kwargs = get_memory_kwargs(mem_type)
        
        results = run_condition(
            memory_type=mem_type,
            memory_kwargs=memory_kwargs,
            num_episodes=num_episodes,
            base_seed=42,
        )
        
        if results:
            config = {
                "num_agents": 3,
                "num_rounds": 10,
                "starting_budget": 20.0,
                "max_contribution": 10,
                "multiplier": 1.8,
                "memory_type": mem_type,
                "memory_kwargs": memory_kwargs,
                "num_episodes": len(results),
                "base_seed": 42,
                "model": "gpt-4o-mini",
                "temperature": 0.7,
            }
            save_results(results_dir, mem_type, config, results)


def run_ablation_experiments(results_dir: Path, num_episodes: int = 10):
    """
    Run ablation experiments to disentangle hybrid memory components.
    
    Conditions:
    - none: Baseline (no memory)
    - structured: Trust table only (no strategy note)
    - hybrid: Trust table + strategy note (full hybrid)
    
    We compare these to understand what component drives hybrid's performance.
    """
    print("\n" + "="*60)
    print("RUNNING ABLATION EXPERIMENTS")
    print("="*60)
    
    conditions = {
        "none": {},
        "structured": {"fair_share_fraction": 0.5},
        "hybrid": {"max_note_words": 20, "fair_share_fraction": 0.5},
    }
    
    for cond_name, memory_kwargs in conditions.items():
        mem_type = cond_name
        
        results = run_condition(
            memory_type=mem_type,
            memory_kwargs=memory_kwargs,
            num_episodes=num_episodes,
            base_seed=100,  # Different seed range for ablations
        )
        
        if results:
            config = {
                "num_agents": 3,
                "num_rounds": 10,
                "starting_budget": 20.0,
                "max_contribution": 10,
                "multiplier": 1.8,
                "memory_type": mem_type,
                "memory_kwargs": memory_kwargs,
                "num_episodes": len(results),
                "base_seed": 100,
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "experiment_type": "ablation",
            }
            save_results(results_dir, f"ablation_{cond_name}", config, results)


def run_alpha_sweep(results_dir: Path, num_episodes: int = 10):
    """
    Run alpha (multiplier) sweep to test robustness.
    
    Tests three alpha values: 1.5, 1.8, 2.1
    For conditions: none, structured, hybrid
    """
    print("\n" + "="*60)
    print("RUNNING ALPHA SWEEP")
    print("="*60)
    
    alphas = [1.5, 1.8, 2.1]
    conditions = ["none", "structured", "hybrid"]
    
    for alpha in alphas:
        print(f"\n--- Alpha = {alpha} ---")
        
        env_config = EnvConfig(
            num_agents=3,
            num_rounds=10,
            starting_budget=20.0,
            max_contribution=10,
            multiplier=alpha,
        )
        
        for mem_type in conditions:
            memory_kwargs = get_memory_kwargs(mem_type)
            
            results = run_condition(
                memory_type=mem_type,
                memory_kwargs=memory_kwargs,
                num_episodes=num_episodes,
                base_seed=200,  # Different seed range for sweeps
                env_config=env_config,
            )
            
            if results:
                config = {
                    "num_agents": 3,
                    "num_rounds": 10,
                    "starting_budget": 20.0,
                    "max_contribution": 10,
                    "multiplier": alpha,
                    "memory_type": mem_type,
                    "memory_kwargs": memory_kwargs,
                    "num_episodes": len(results),
                    "base_seed": 200,
                    "model": "gpt-4o-mini",
                    "temperature": 0.7,
                    "experiment_type": "alpha_sweep",
                }
                save_results(results_dir, f"sweep_alpha{alpha}_{mem_type}", config, results)


def main():
    """Run all experiments."""
    # Verify API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file")
        sys.exit(1)
    
    print(f"API key found (starts with: {api_key[:8]}...)")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Run experiments
    try:
        # Main experiments: 15 episodes per condition
        run_main_experiments(results_dir, num_episodes=15)
        
        # Ablation experiments: 10 episodes per condition
        run_ablation_experiments(results_dir, num_episodes=10)
        
        # Alpha sweep: 10 episodes per condition per alpha
        run_alpha_sweep(results_dir, num_episodes=10)
        
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user.")
    except Exception as e:
        print(f"\n\nExperiments failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    
    # Print summary
    print("\nResults saved to:", results_dir)
    for json_file in sorted(results_dir.glob("*.json")):
        print(f"  - {json_file.name}")


if __name__ == "__main__":
    main()
