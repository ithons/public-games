"""
Base Memory Interface

All memory modules implement this interface for pluggable memory representations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RoundInfo:
    """Information about a completed round."""
    round_index: int
    agent_id: int
    contributions: list[int]  # All agents' contributions
    pot: float
    payoffs: list[float]  # All agents' payoffs
    own_budget: float
    own_cumulative_payoff: float


@dataclass
class MemoryState:
    """
    Container for memory state that can be any structure.
    Wraps the actual state with metadata for analysis.
    """
    state: Any
    token_estimate: int = 0  # Estimated tokens when rendered
    
    def __repr__(self) -> str:
        return f"MemoryState(tokens≈{self.token_estimate}, state={type(self.state).__name__})"


class BaseMemory(ABC):
    """
    Abstract base class for memory modules.
    
    Each memory module defines:
    1. How to initialize memory state for an agent
    2. How to update memory after each round
    3. How to render memory for inclusion in LLM prompts
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this memory type."""
        pass
    
    @abstractmethod
    def init_state(self, agent_id: int, num_agents: int) -> MemoryState:
        """
        Initialize memory state for a new episode.
        
        Args:
            agent_id: The agent's ID (0-indexed)
            num_agents: Total number of agents in the game
            
        Returns:
            Initial MemoryState
        """
        pass
    
    @abstractmethod
    def update(
        self,
        state: MemoryState,
        round_info: RoundInfo,
        llm_response: Optional[str] = None,
    ) -> MemoryState:
        """
        Update memory state after a round completes.
        
        Args:
            state: Current memory state
            round_info: Information about the completed round
            llm_response: Optional raw LLM response (for summary updates)
            
        Returns:
            Updated MemoryState
        """
        pass
    
    @abstractmethod
    def render_for_prompt(self, state: MemoryState) -> str:
        """
        Render memory state as text for inclusion in LLM prompts.
        
        Args:
            state: Current memory state
            
        Returns:
            Text representation of memory
        """
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimate of token count for a string.
        Uses ~4 characters per token as approximation.
        """
        return len(text) // 4 + 1


def format_contributions(contributions: list[int], own_agent_id: int) -> str:
    """Format contributions list with own agent highlighted."""
    parts = []
    for i, c in enumerate(contributions):
        if i == own_agent_id:
            parts.append(f"You: {c}")
        else:
            parts.append(f"Agent {i}: {c}")
    return ", ".join(parts)

