"""
No Memory Module

Agent receives only current state and last round's contributions.
No historical information is retained.
"""

from typing import Optional
from memory.base_memory import BaseMemory, MemoryState, RoundInfo


class NoMemory(BaseMemory):
    """
    Baseline memory: no historical information retained.
    Agent only knows current round and last round's outcomes.
    """
    
    @property
    def name(self) -> str:
        return "No Memory"
    
    def __init__(self) -> None:
        pass
    
    def init_state(self, agent_id: int, num_agents: int) -> MemoryState:
        return MemoryState(
            state={"agent_id": agent_id, "num_agents": num_agents},
            token_estimate=5,
        )
    
    def update(
        self,
        state: MemoryState,
        round_info: RoundInfo,
        llm_response: Optional[str] = None,
    ) -> MemoryState:
        # No memory to update - state remains the same
        return state
    
    def render_for_prompt(self, state: MemoryState) -> str:
        return "[No memory of previous rounds available]"

