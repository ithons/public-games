"""
Experiment Runner

CLI and programmatic interface for running public goods experiments
with different memory conditions.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from environment.public_goods_env import PublicGoodsEnv, EnvConfig, RoundObservation
from memory import create_memory, BaseMemory, MEMORY_REGISTRY
from memory.base_memory import MemoryState, RoundInfo
from agents.base_policy import AgentPolicy, AgentAction
from agents.llm_policy import LLMPolicy, MockLLMBackend, OpenAIBackend, LLMBackend


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    # Environment parameters
    num_agents: int = 3
    num_rounds: int = 10
    starting_budget: float = 20.0
    max_contribution: int = 10
    multiplier: float = 1.8
    
    # Memory parameters
    memory_type: str = "none"
    memory_kwargs: dict = None
    
    # Experiment parameters
    num_episodes: int = 30
    base_seed: int = 42
    
    # LLM parameters
    use_real_llm: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    mock_strategy: str = "cooperative"
    
    # Output
    results_dir: str = "results"
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        if self.memory_kwargs is None:
            self.memory_kwargs = {}
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeResult:
    """Results from a single episode."""
    episode_id: int
    seed: int
    total_rounds: int
    
    # Per-round data
    contributions: list[list[int]]  # [round][agent]
    payoffs: list[list[float]]  # [round][agent]
    budgets: list[list[float]]  # [round][agent] (after each round)
    pots: list[float]  # [round]
    
    # Aggregate metrics
    total_welfare: float  # Sum of all payoffs
    final_budgets: list[float]
    mean_contribution: float
    contribution_variance: float
    
    # Token metrics
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    memory_token_estimates: list[int] = None  # [round]
    
    def to_dict(self) -> dict:
        return asdict(self)


def create_backend(config: ExperimentConfig) -> LLMBackend:
    """Create appropriate LLM backend based on config."""
    if config.use_real_llm:
        return OpenAIBackend(
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
    else:
        return MockLLMBackend(strategy=config.mock_strategy)


def run_episode(
    env: PublicGoodsEnv,
    agents: list[LLMPolicy],
    memories: list[BaseMemory],
    memory_states: list[MemoryState],
    seed: int,
    episode_id: int,
) -> EpisodeResult:
    """Run a single episode and collect results."""
    
    observations = env.reset(seed=seed)
    
    contributions_log: list[list[int]] = []
    payoffs_log: list[list[float]] = []
    budgets_log: list[list[float]] = []
    pots_log: list[float] = []
    memory_tokens_log: list[int] = []
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    done = False
    while not done:
        # Collect actions from all agents
        actions: list[AgentAction] = []
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
        if not done:  # Don't update after final round
            round_idx = env.current_round - 1
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
                
                # Extract LLM response for summary updates
                llm_response = action.raw_response if hasattr(action, "raw_response") else None
                memory_states[i] = memory.update(memory_states[i], round_info, llm_response)
            
            memory_tokens_log.append(sum(ms.token_estimate for ms in memory_states))
    
    # Compute aggregate metrics
    all_contributions = [c for round_contribs in contributions_log for c in round_contribs]
    mean_contribution = sum(all_contributions) / len(all_contributions) if all_contributions else 0
    variance = sum((c - mean_contribution) ** 2 for c in all_contributions) / len(all_contributions) if all_contributions else 0
    
    total_welfare = sum(sum(round_payoffs) for round_payoffs in payoffs_log)
    final_budgets = budgets_log[-1] if budgets_log else []
    
    return EpisodeResult(
        episode_id=episode_id,
        seed=seed,
        total_rounds=env.config.num_rounds,
        contributions=contributions_log,
        payoffs=payoffs_log,
        budgets=budgets_log,
        pots=pots_log,
        total_welfare=total_welfare,
        final_budgets=final_budgets,
        mean_contribution=mean_contribution,
        contribution_variance=variance,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        memory_token_estimates=memory_tokens_log,
    )


def run_experiment(config: ExperimentConfig) -> list[EpisodeResult]:
    """Run a complete experiment with the given configuration."""
    
    # Create environment
    env_config = EnvConfig(
        num_agents=config.num_agents,
        num_rounds=config.num_rounds,
        starting_budget=config.starting_budget,
        max_contribution=config.max_contribution,
        multiplier=config.multiplier,
    )
    env = PublicGoodsEnv(env_config)
    
    # Create backend (shared across agents)
    backend = create_backend(config)
    
    # Run episodes
    results: list[EpisodeResult] = []
    
    for ep in range(config.num_episodes):
        seed = config.base_seed + ep
        
        # Create fresh memories and agents for each episode
        memories: list[BaseMemory] = []
        memory_states: list[MemoryState] = []
        agents: list[LLMPolicy] = []
        
        for agent_id in range(config.num_agents):
            memory = create_memory(config.memory_type, **config.memory_kwargs)
            memories.append(memory)
            memory_states.append(memory.init_state(agent_id, config.num_agents))
            
            agent = LLMPolicy(
                backend=backend,
                memory=memory,
                num_agents=config.num_agents,
                multiplier=config.multiplier,
            )
            agent.reset()
            agents.append(agent)
        
        # Run episode
        episode_result = run_episode(
            env=env,
            agents=agents,
            memories=memories,
            memory_states=memory_states,
            seed=seed,
            episode_id=ep,
        )
        results.append(episode_result)
        
        if (ep + 1) % 10 == 0:
            print(f"  Completed episode {ep + 1}/{config.num_episodes}")
    
    return results


def save_results(
    config: ExperimentConfig,
    results: list[EpisodeResult],
    output_dir: Path,
) -> Path:
    """Save experiment results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config.experiment_name or f"{config.memory_type}_{timestamp}"
    output_file = output_dir / f"{name}.json"
    
    data = {
        "config": config.to_dict(),
        "results": [r.to_dict() for r in results],
        "timestamp": timestamp,
    }
    
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    
    return output_file


def run_all_conditions(
    base_config: ExperimentConfig,
    memory_types: Optional[list[str]] = None,
) -> dict[str, list[EpisodeResult]]:
    """Run experiments for all memory conditions."""
    
    if memory_types is None:
        memory_types = list(MEMORY_REGISTRY.keys())
    
    all_results: dict[str, list[EpisodeResult]] = {}
    
    for mem_type in memory_types:
        print(f"\n=== Running {mem_type} memory condition ===")
        
        config = ExperimentConfig(
            num_agents=base_config.num_agents,
            num_rounds=base_config.num_rounds,
            starting_budget=base_config.starting_budget,
            max_contribution=base_config.max_contribution,
            multiplier=base_config.multiplier,
            memory_type=mem_type,
            memory_kwargs=_get_memory_kwargs(mem_type),
            num_episodes=base_config.num_episodes,
            base_seed=base_config.base_seed,
            use_real_llm=base_config.use_real_llm,
            llm_model=base_config.llm_model,
            llm_temperature=base_config.llm_temperature,
            mock_strategy=base_config.mock_strategy,
            results_dir=base_config.results_dir,
            experiment_name=f"{mem_type}",
        )
        
        results = run_experiment(config)
        all_results[mem_type] = results
        
        # Save results
        output_dir = Path(config.results_dir)
        output_file = save_results(config, results, output_dir)
        print(f"  Saved to {output_file}")
    
    return all_results


def _get_memory_kwargs(memory_type: str) -> dict:
    """Get default kwargs for each memory type."""
    defaults = {
        "none": {},
        "full_history": {"window_size": 5},
        "summary": {"max_words": 50},
        "structured": {"fair_share_fraction": 0.5},
        "hybrid": {"max_note_words": 20, "fair_share_fraction": 0.5},
    }
    return defaults.get(memory_type, {})


@click.command()
@click.option("--memory-type", "-m", default="none", help="Memory type to use")
@click.option("--all-conditions", is_flag=True, help="Run all memory conditions")
@click.option("--episodes", "-e", default=30, help="Number of episodes")
@click.option("--rounds", "-r", default=10, help="Rounds per episode")
@click.option("--agents", "-n", default=3, help="Number of agents")
@click.option("--seed", "-s", default=42, help="Base random seed")
@click.option("--use-real-llm", is_flag=True, help="Use real OpenAI API")
@click.option("--model", default="gpt-4o-mini", help="LLM model to use")
@click.option("--results-dir", "-o", default="results", help="Output directory")
def main(
    memory_type: str,
    all_conditions: bool,
    episodes: int,
    rounds: int,
    agents: int,
    seed: int,
    use_real_llm: bool,
    model: str,
    results_dir: str,
):
    """Run public goods game experiments with LLM agents."""
    
    base_config = ExperimentConfig(
        num_agents=agents,
        num_rounds=rounds,
        num_episodes=episodes,
        base_seed=seed,
        use_real_llm=use_real_llm,
        llm_model=model,
        results_dir=results_dir,
    )
    
    if all_conditions:
        print("Running all memory conditions...")
        run_all_conditions(base_config)
    else:
        print(f"Running {memory_type} memory condition...")
        config = ExperimentConfig(
            **{**base_config.to_dict(), "memory_type": memory_type}
        )
        config.memory_kwargs = _get_memory_kwargs(memory_type)
        results = run_experiment(config)
        
        output_dir = Path(results_dir)
        output_file = save_results(config, results, output_dir)
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()

