"""
Full History Memory Module

Maintains a sliding window of detailed round-by-round history.
"""

from dataclasses import dataclass, field
from typing import Optional
from memory.base_memory import BaseMemory, MemoryState, RoundInfo, format_contributions


@dataclass
class HistoryEntry:
    """Single round entry in history."""
    round_index: int
    contributions: list[int]
    pot: float
    payoffs: list[float]
    own_contribution: int
    own_payoff: float


@dataclass  
class FullHistoryState:
    """State for full history memory."""
    agent_id: int
    num_agents: int
    history: list[HistoryEntry] = field(default_factory=list)
    window_size: int = 10  # Maximum rounds to retain


class FullHistoryMemory(BaseMemory):
    """
    Full history memory: keeps detailed log of last k rounds.
    
    For each round, stores:
    - All agents' contributions
    - Total pot and payoffs
    - Own contribution and payoff
    """
    
    @property
    def name(self) -> str:
        return f"Full History (k={self.window_size})"
    
    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
    
    def init_state(self, agent_id: int, num_agents: int) -> MemoryState:
        state = FullHistoryState(
            agent_id=agent_id,
            num_agents=num_agents,
            history=[],
            window_size=self.window_size,
        )
        return MemoryState(state=state, token_estimate=10)
    
    def update(
        self,
        state: MemoryState,
        round_info: RoundInfo,
        llm_response: Optional[str] = None,
    ) -> MemoryState:
        hist_state: FullHistoryState = state.state
        
        entry = HistoryEntry(
            round_index=round_info.round_index,
            contributions=round_info.contributions.copy(),
            pot=round_info.pot,
            payoffs=round_info.payoffs.copy(),
            own_contribution=round_info.contributions[hist_state.agent_id],
            own_payoff=round_info.payoffs[hist_state.agent_id],
        )
        
        hist_state.history.append(entry)
        
        # Truncate to window size
        if len(hist_state.history) > hist_state.window_size:
            hist_state.history = hist_state.history[-hist_state.window_size:]
        
        # Re-render to estimate tokens
        rendered = self.render_for_prompt(state)
        return MemoryState(state=hist_state, token_estimate=self.estimate_tokens(rendered))
    
    def render_for_prompt(self, state: MemoryState) -> str:
        hist_state: FullHistoryState = state.state
        
        if not hist_state.history:
            return "[No previous rounds yet]"
        
        lines = [f"=== History of last {len(hist_state.history)} round(s) ==="]
        
        for entry in hist_state.history:
            lines.append(f"\nRound {entry.round_index + 1}:")
            lines.append(f"  Contributions: {format_contributions(entry.contributions, hist_state.agent_id)}")
            lines.append(f"  Total pot (after multiplier): {entry.pot:.1f}")
            lines.append(f"  Your contribution: {entry.own_contribution}, Your payoff: {entry.own_payoff:+.1f}")
        
        return "\n".join(lines)

