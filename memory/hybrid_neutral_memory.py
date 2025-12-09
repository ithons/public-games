"""
Hybrid Neutral Memory Module

Identical to Hybrid memory but with a neutrally-initialized strategy note.
Used for ablation to distinguish articulation effects from priming effects.
"""

from memory.hybrid_memory import HybridMemory, HybridState
from memory.base_memory import MemoryState
from memory.structured_memory import StructuredState, AgentRecord


class HybridNeutralMemory(HybridMemory):
    """
    Hybrid memory with neutral strategy note initialization.
    
    Identical to HybridMemory except:
    - Initial note: "Decide each round based on the situation and observed outcomes."
    - (vs cooperative: "Start by cooperating to establish trust.")
    
    This allows testing whether the strategy note effect comes from:
    (a) The act of articulating any strategy (articulation effect)
    (b) The cooperative content of the initial seed (priming effect)
    """
    
    NEUTRAL_STRATEGY_NOTE = "Decide each round based on the situation and observed outcomes."
    
    @property
    def name(self) -> str:
        return f"Hybrid-Neutral (table + {self.max_note_words}w neutral note)"
    
    def init_state(self, agent_id: int, num_agents: int) -> MemoryState:
        structured_state = StructuredState(
            agent_id=agent_id,
            num_agents=num_agents,
            records={i: AgentRecord(agent_id=i) for i in range(num_agents)},
            fair_share_threshold=self.fair_share_fraction,
        )
        
        state = HybridState(
            structured=structured_state,
            strategy_note=self.NEUTRAL_STRATEGY_NOTE,  # Neutral instead of cooperative
            max_note_words=self.max_note_words,
        )
        
        return MemoryState(state=state, token_estimate=30)
