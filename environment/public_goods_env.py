"""
Public Goods Game Environment

A repeated public-goods game where N agents decide how much to contribute
to a shared pot each round. The pot is multiplied and redistributed equally,
creating tension between individual incentives (defect) and collective welfare.
"""

from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class EnvConfig:
    """Configuration for the public goods environment."""
    num_agents: int = 3
    num_rounds: int = 10
    starting_budget: float = 20.0
    max_contribution: int = 10
    multiplier: float = 1.8  # α: pot multiplier (should be in (1, num_agents) for dilemma)
    
    def __post_init__(self) -> None:
        if self.multiplier <= 1:
            raise ValueError("Multiplier must be > 1 for cooperation to be beneficial")
        if self.multiplier >= self.num_agents:
            raise ValueError("Multiplier must be < num_agents for defection to be tempting")


@dataclass
class RoundObservation:
    """Observation provided to each agent at the start of a round."""
    round_index: int
    agent_id: int
    budget: float
    cumulative_payoff: float
    last_round_contributions: Optional[list[int]]  # None for round 0
    last_round_pot: Optional[float]
    last_round_payoffs: Optional[list[float]]
    

@dataclass
class StepResult:
    """Result of a single environment step."""
    observations: list[RoundObservation]
    rewards: list[float]
    done: bool
    info: dict


@dataclass
class EpisodeLog:
    """Complete log of an episode for analysis."""
    config: EnvConfig
    seed: int
    rounds: list[dict] = field(default_factory=list)
    
    def add_round(
        self,
        round_index: int,
        contributions: list[int],
        pot: float,
        payoffs: list[float],
        budgets: list[float],
    ) -> None:
        self.rounds.append({
            "round": round_index,
            "contributions": contributions.copy(),
            "pot": pot,
            "payoffs": payoffs.copy(),
            "budgets": budgets.copy(),
        })
    
    def to_dict(self) -> dict:
        return {
            "config": {
                "num_agents": self.config.num_agents,
                "num_rounds": self.config.num_rounds,
                "starting_budget": self.config.starting_budget,
                "max_contribution": self.config.max_contribution,
                "multiplier": self.config.multiplier,
            },
            "seed": self.seed,
            "rounds": self.rounds,
        }


class PublicGoodsEnv:
    """
    Repeated Public Goods Game Environment.
    
    At each round:
    1. Agents observe their budget, cumulative payoff, and last round's outcomes
    2. Agents simultaneously choose contributions (0 to max_contribution, ≤ budget)
    3. Contributions form a pot, multiplied by α, then split equally
    4. Each agent's budget is updated: budget -= contribution + share
    """
    
    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self._rng: random.Random = random.Random()
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Initialize/reset internal state variables."""
        self._current_round = 0
        self._budgets: list[float] = []
        self._cumulative_payoffs: list[float] = []
        self._last_contributions: Optional[list[int]] = None
        self._last_pot: Optional[float] = None
        self._last_payoffs: Optional[list[float]] = None
        self._episode_log: Optional[EpisodeLog] = None
        self._seed: int = 0
    
    def reset(self, seed: Optional[int] = None) -> list[RoundObservation]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observations for all agents
        """
        self._reset_state()
        
        if seed is not None:
            self._seed = seed
            self._rng.seed(seed)
        else:
            self._seed = self._rng.randint(0, 2**31 - 1)
            self._rng.seed(self._seed)
        
        self._budgets = [self.config.starting_budget] * self.config.num_agents
        self._cumulative_payoffs = [0.0] * self.config.num_agents
        self._episode_log = EpisodeLog(config=self.config, seed=self._seed)
        
        return self._get_observations()
    
    def _get_observations(self) -> list[RoundObservation]:
        """Generate observations for all agents."""
        return [
            RoundObservation(
                round_index=self._current_round,
                agent_id=i,
                budget=self._budgets[i],
                cumulative_payoff=self._cumulative_payoffs[i],
                last_round_contributions=self._last_contributions,
                last_round_pot=self._last_pot,
                last_round_payoffs=self._last_payoffs,
            )
            for i in range(self.config.num_agents)
        ]
    
    def step(self, contributions: list[int]) -> StepResult:
        """
        Execute one round of the game.
        
        Args:
            contributions: List of contribution amounts from each agent
            
        Returns:
            StepResult with new observations, rewards, done flag, and info
        """
        if len(contributions) != self.config.num_agents:
            raise ValueError(
                f"Expected {self.config.num_agents} contributions, got {len(contributions)}"
            )
        
        # Validate and clip contributions
        validated_contributions: list[int] = []
        for i, c in enumerate(contributions):
            # Contribution must be non-negative, at most max_contribution, and within budget
            valid_c = max(0, min(c, self.config.max_contribution, int(self._budgets[i])))
            validated_contributions.append(valid_c)
        
        # Calculate pot and payoffs
        pot = sum(validated_contributions)
        multiplied_pot = pot * self.config.multiplier
        share = multiplied_pot / self.config.num_agents
        
        # Calculate individual payoffs and update state
        rewards: list[float] = []
        for i in range(self.config.num_agents):
            # Net gain: share received minus contribution made
            reward = share - validated_contributions[i]
            rewards.append(reward)
            
            # Update budget: subtract contribution, add share
            self._budgets[i] = self._budgets[i] - validated_contributions[i] + share
            self._cumulative_payoffs[i] += reward
        
        # Log this round
        if self._episode_log is not None:
            self._episode_log.add_round(
                round_index=self._current_round,
                contributions=validated_contributions,
                pot=multiplied_pot,
                payoffs=rewards,
                budgets=self._budgets,
            )
        
        # Update state for next round
        self._last_contributions = validated_contributions
        self._last_pot = multiplied_pot
        self._last_payoffs = rewards
        self._current_round += 1
        
        done = self._current_round >= self.config.num_rounds
        
        return StepResult(
            observations=self._get_observations(),
            rewards=rewards,
            done=done,
            info={
                "round": self._current_round - 1,
                "contributions": validated_contributions,
                "pot": multiplied_pot,
                "share": share,
            },
        )
    
    def get_episode_log(self) -> Optional[EpisodeLog]:
        """Return the episode log for analysis."""
        return self._episode_log
    
    @property
    def current_round(self) -> int:
        return self._current_round
    
    @property
    def budgets(self) -> list[float]:
        return self._budgets.copy()
    
    @property
    def cumulative_payoffs(self) -> list[float]:
        return self._cumulative_payoffs.copy()


def compute_nash_equilibrium(config: EnvConfig) -> dict:
    """
    Compute theoretical Nash equilibrium for the public goods game.
    
    In a one-shot public goods game with linear payoffs:
    - If α/n < 1: Nash equilibrium is to contribute 0 (dominant strategy to defect)
    - Social optimum is always to contribute max
    
    Returns dict with theoretical predictions.
    """
    marginal_return = config.multiplier / config.num_agents
    
    return {
        "marginal_return_per_unit": marginal_return,
        "nash_contribution": 0 if marginal_return < 1 else config.max_contribution,
        "social_optimal_contribution": config.max_contribution,
        "is_social_dilemma": marginal_return < 1 < config.multiplier,
        "cooperation_benefit": config.multiplier - 1,  # Extra value created per unit contributed
    }

