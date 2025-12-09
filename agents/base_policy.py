"""
Base Agent Policy Interface

Defines the abstract interface for all agent policies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from environment.public_goods_env import RoundObservation
from memory.base_memory import MemoryState


@dataclass
class AgentAction:
    """Result of an agent's decision."""
    contribution: int
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class AgentPolicy(ABC):
    """
    Abstract base class for agent policies.
    
    Defines the interface for agents that make contribution decisions
    based on observations and memory.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this policy."""
        pass
    
    @abstractmethod
    def act(
        self,
        observation: RoundObservation,
        memory_state: MemoryState,
        max_contribution: int,
    ) -> AgentAction:
        """
        Choose a contribution based on current observation and memory.
        
        Args:
            observation: Current round observation
            memory_state: Agent's current memory state
            max_contribution: Maximum allowed contribution
            
        Returns:
            AgentAction with chosen contribution and metadata
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state for a new episode."""
        pass


class RandomPolicy(AgentPolicy):
    """Simple random baseline policy for testing."""
    
    def __init__(self, seed: Optional[int] = None) -> None:
        import random
        self._rng = random.Random(seed)
    
    @property
    def name(self) -> str:
        return "Random"
    
    def act(
        self,
        observation: RoundObservation,
        memory_state: MemoryState,
        max_contribution: int,
    ) -> AgentAction:
        max_possible = min(max_contribution, int(observation.budget))
        contribution = self._rng.randint(0, max_possible)
        return AgentAction(
            contribution=contribution,
            reasoning="Random choice",
        )
    
    def reset(self) -> None:
        pass


class FixedPolicy(AgentPolicy):
    """Fixed contribution policy for testing."""
    
    def __init__(self, contribution: int) -> None:
        self._contribution = contribution
    
    @property
    def name(self) -> str:
        return f"Fixed({self._contribution})"
    
    def act(
        self,
        observation: RoundObservation,
        memory_state: MemoryState,
        max_contribution: int,
    ) -> AgentAction:
        actual = min(self._contribution, max_contribution, int(observation.budget))
        return AgentAction(
            contribution=actual,
            reasoning=f"Fixed contribution of {self._contribution}",
        )
    
    def reset(self) -> None:
        pass


class TitForTatPolicy(AgentPolicy):
    """
    Tit-for-tat style policy: cooperate initially, then mirror average behavior.
    """
    
    def __init__(self, initial_contribution: int = 5) -> None:
        self._initial = initial_contribution
        self._round = 0
    
    @property
    def name(self) -> str:
        return "TitForTat"
    
    def act(
        self,
        observation: RoundObservation,
        memory_state: MemoryState,
        max_contribution: int,
    ) -> AgentAction:
        max_possible = min(max_contribution, int(observation.budget))
        
        if self._round == 0 or observation.last_round_contributions is None:
            contribution = min(self._initial, max_possible)
            reasoning = "First round: cooperate"
        else:
            # Match average of others' contributions
            others = [
                c for i, c in enumerate(observation.last_round_contributions)
                if i != observation.agent_id
            ]
            avg_others = sum(others) / len(others) if others else self._initial
            contribution = min(int(round(avg_others)), max_possible)
            reasoning = f"Matching others' average: {avg_others:.1f}"
        
        self._round += 1
        return AgentAction(contribution=contribution, reasoning=reasoning)
    
    def reset(self) -> None:
        self._round = 0

