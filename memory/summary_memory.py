"""
Summary Memory Module

Maintains a rolling natural-language summary updated by the LLM each round.
"""

from dataclasses import dataclass
from typing import Optional
from memory.base_memory import BaseMemory, MemoryState, RoundInfo, format_contributions


@dataclass
class SummaryState:
    """State for summary memory."""
    agent_id: int
    num_agents: int
    summary: str
    max_words: int
    last_round_context: str  # Context to help LLM update summary


class SummaryMemory(BaseMemory):
    """
    Summary memory: LLM-generated rolling summary of game history.
    
    The summary is updated each round by asking the LLM to incorporate
    the latest round's events while staying within a word limit.
    """
    
    @property
    def name(self) -> str:
        return f"Summary ({self.max_words} words)"
    
    def __init__(self, max_words: int = 50) -> None:
        self.max_words = max_words
    
    def init_state(self, agent_id: int, num_agents: int) -> MemoryState:
        state = SummaryState(
            agent_id=agent_id,
            num_agents=num_agents,
            summary="No history yet. This is the first round.",
            max_words=self.max_words,
            last_round_context="",
        )
        return MemoryState(state=state, token_estimate=15)
    
    def update(
        self,
        state: MemoryState,
        round_info: RoundInfo,
        llm_response: Optional[str] = None,
    ) -> MemoryState:
        sum_state: SummaryState = state.state
        
        # Build context for what happened this round
        contributions_str = format_contributions(round_info.contributions, sum_state.agent_id)
        sum_state.last_round_context = (
            f"Round {round_info.round_index + 1}: {contributions_str}. "
            f"Pot={round_info.pot:.1f}, Your payoff={round_info.payoffs[sum_state.agent_id]:+.1f}"
        )
        
        # If LLM provided an updated summary, use it
        if llm_response:
            # Extract summary from response (agent should output summary update)
            sum_state.summary = self._extract_summary(llm_response, sum_state.summary)
        else:
            # Fallback: append to existing summary (will be truncated in render)
            sum_state.summary = f"{sum_state.summary} {sum_state.last_round_context}"
        
        rendered = self.render_for_prompt(state)
        return MemoryState(state=sum_state, token_estimate=self.estimate_tokens(rendered))
    
    def _extract_summary(self, llm_response: str, fallback: str) -> str:
        """Extract summary update from LLM response."""
        # Look for summary in response
        response_lower = llm_response.lower()
        
        # Try to find explicit summary markers
        markers = ["updated summary:", "summary:", "memory update:"]
        for marker in markers:
            if marker in response_lower:
                idx = response_lower.index(marker) + len(marker)
                # Take text after marker until end or next section
                summary_text = llm_response[idx:].strip()
                # Truncate at reasonable length
                words = summary_text.split()[:self.max_words]
                return " ".join(words)
        
        return fallback
    
    def render_for_prompt(self, state: MemoryState) -> str:
        sum_state: SummaryState = state.state
        
        lines = [
            "=== Your Memory Summary ===",
            sum_state.summary,
        ]
        
        if sum_state.last_round_context:
            lines.append(f"\n[Latest: {sum_state.last_round_context}]")
        
        return "\n".join(lines)
    
    def get_summary_update_prompt(self, state: MemoryState) -> str:
        """
        Generate prompt asking LLM to update the summary.
        Called by the agent to get summary update instruction.
        """
        sum_state: SummaryState = state.state
        
        return (
            f"\n\nAfter making your decision, update your memory summary. "
            f"Current summary: \"{sum_state.summary}\"\n"
            f"New information: {sum_state.last_round_context}\n"
            f"Provide an updated summary in at most {self.max_words} words that captures "
            f"the most important patterns for future decisions. "
            f"Format: 'Updated summary: <your summary>'"
        )

