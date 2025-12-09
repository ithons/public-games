"""
Hybrid Memory Module

Combines structured trust table with a short LLM-generated strategy note.
"""

from dataclasses import dataclass
from typing import Optional
from memory.base_memory import BaseMemory, MemoryState, RoundInfo
from memory.structured_memory import StructuredState, StructuredMemory, AgentRecord


@dataclass
class HybridState:
    """State for hybrid memory."""
    structured: StructuredState
    strategy_note: str
    max_note_words: int


class HybridMemory(BaseMemory):
    """
    Hybrid memory: structured trust table + short strategy note.
    
    Combines:
    - Numerical trust table (like StructuredMemory)
    - One-sentence strategy note updated by LLM
    
    The strategy note allows the agent to capture qualitative insights
    while the table provides precise numerical context.
    """
    
    @property
    def name(self) -> str:
        return f"Hybrid (table + {self.max_note_words}w note)"
    
    def __init__(self, max_note_words: int = 20, fair_share_fraction: float = 0.5) -> None:
        self.max_note_words = max_note_words
        self.fair_share_fraction = fair_share_fraction
        self._structured_helper = StructuredMemory(fair_share_fraction)
    
    def init_state(self, agent_id: int, num_agents: int) -> MemoryState:
        structured_state = StructuredState(
            agent_id=agent_id,
            num_agents=num_agents,
            records={i: AgentRecord(agent_id=i) for i in range(num_agents)},
            fair_share_threshold=self.fair_share_fraction,
        )
        
        state = HybridState(
            structured=structured_state,
            strategy_note="Start by cooperating to establish trust.",
            max_note_words=self.max_note_words,
        )
        
        return MemoryState(state=state, token_estimate=30)
    
    def update(
        self,
        state: MemoryState,
        round_info: RoundInfo,
        llm_response: Optional[str] = None,
    ) -> MemoryState:
        hybrid_state: HybridState = state.state
        
        # Update structured part using helper logic
        temp_state = MemoryState(state=hybrid_state.structured, token_estimate=0)
        updated_struct = self._structured_helper.update(temp_state, round_info, None)
        hybrid_state.structured = updated_struct.state
        
        # Update strategy note if LLM provided one
        if llm_response:
            new_note = self._extract_strategy_note(llm_response)
            if new_note:
                hybrid_state.strategy_note = new_note
        
        rendered = self.render_for_prompt(state)
        return MemoryState(state=hybrid_state, token_estimate=self.estimate_tokens(rendered))
    
    def _extract_strategy_note(self, llm_response: str) -> Optional[str]:
        """Extract strategy note from LLM response."""
        response_lower = llm_response.lower()
        
        markers = ["strategy note:", "note:", "strategy:"]
        for marker in markers:
            if marker in response_lower:
                idx = response_lower.index(marker) + len(marker)
                note_text = llm_response[idx:].strip()
                # Take first sentence or up to max words
                words = note_text.split()[:self.max_note_words]
                note = " ".join(words)
                # Truncate at sentence end if possible
                for end in [".", "!", "?"]:
                    if end in note:
                        note = note[:note.index(end) + 1]
                        break
                return note
        
        return None
    
    def render_for_prompt(self, state: MemoryState) -> str:
        hybrid_state: HybridState = state.state
        
        # Render structured table
        temp_state = MemoryState(state=hybrid_state.structured, token_estimate=0)
        table_str = self._structured_helper.render_for_prompt(temp_state)
        
        lines = [
            table_str,
            "",
            f"=== Your Strategy Note ===",
            hybrid_state.strategy_note,
        ]
        
        return "\n".join(lines)
    
    def get_note_update_prompt(self) -> str:
        """Generate prompt asking LLM to update strategy note."""
        return (
            f"\n\nOptionally update your strategy note (max {self.max_note_words} words) "
            f"based on what you've learned. Format: 'Strategy note: <your note>'"
        )

