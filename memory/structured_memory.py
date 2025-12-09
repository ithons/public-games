"""
Structured Memory Module

Maintains a compact numerical trust table with cooperation scores per agent.
"""

from dataclasses import dataclass, field
from typing import Optional
from memory.base_memory import BaseMemory, MemoryState, RoundInfo


@dataclass
class AgentRecord:
    """Record for a single other agent."""
    agent_id: int
    cooperation_score: float = 0.5  # [0, 1] scale
    total_contributions: int = 0
    contribution_count: int = 0
    defection_count: int = 0  # Rounds where they contributed less than fair share
    last_contribution: int = 0
    
    @property
    def avg_contribution(self) -> float:
        if self.contribution_count == 0:
            return 0.0
        return self.total_contributions / self.contribution_count


@dataclass
class StructuredState:
    """State for structured memory."""
    agent_id: int
    num_agents: int
    records: dict[int, AgentRecord] = field(default_factory=dict)
    fair_share_threshold: float = 0.5  # Fraction of max contribution considered "fair"
    max_contribution: int = 10
    rounds_seen: int = 0
    
    # Decay and update parameters
    score_update_rate: float = 0.3  # How much new info affects score


class StructuredMemory(BaseMemory):
    """
    Structured memory: compact trust table per agent.
    
    For each other agent, tracks:
    - Cooperation score (exponential moving average of cooperation)
    - Last contribution
    - Defection count (times they contributed below fair share)
    
    Updates are purely heuristic, no learning involved.
    """
    
    @property
    def name(self) -> str:
        return "Structured Trust Table"
    
    def __init__(self, fair_share_fraction: float = 0.5) -> None:
        self.fair_share_fraction = fair_share_fraction
    
    def init_state(self, agent_id: int, num_agents: int) -> MemoryState:
        state = StructuredState(
            agent_id=agent_id,
            num_agents=num_agents,
            records={},
            fair_share_threshold=self.fair_share_fraction,
        )
        
        # Initialize records for all agents (including self for completeness)
        for i in range(num_agents):
            state.records[i] = AgentRecord(agent_id=i)
        
        return MemoryState(state=state, token_estimate=20)
    
    def update(
        self,
        state: MemoryState,
        round_info: RoundInfo,
        llm_response: Optional[str] = None,
    ) -> MemoryState:
        struct_state: StructuredState = state.state
        struct_state.rounds_seen += 1
        struct_state.max_contribution = max(
            struct_state.max_contribution, 
            max(round_info.contributions) if round_info.contributions else 10
        )
        
        # Calculate what "fair share" means this round
        fair_contribution = struct_state.max_contribution * struct_state.fair_share_threshold
        
        # Calculate average contribution to detect relative defection
        avg_contribution = sum(round_info.contributions) / len(round_info.contributions)
        
        # Update each agent's record
        for i, contribution in enumerate(round_info.contributions):
            record = struct_state.records[i]
            record.last_contribution = contribution
            record.total_contributions += contribution
            record.contribution_count += 1
            
            # Check for defection (contributing significantly less than others or fair share)
            is_defection = contribution < fair_contribution and contribution < avg_contribution * 0.7
            if is_defection:
                record.defection_count += 1
            
            # Update cooperation score with exponential moving average
            # Normalize contribution to [0, 1]
            normalized = contribution / struct_state.max_contribution if struct_state.max_contribution > 0 else 0.5
            record.cooperation_score = (
                (1 - struct_state.score_update_rate) * record.cooperation_score +
                struct_state.score_update_rate * normalized
            )
            # Clamp to [0, 1]
            record.cooperation_score = max(0.0, min(1.0, record.cooperation_score))
        
        rendered = self.render_for_prompt(state)
        return MemoryState(state=struct_state, token_estimate=self.estimate_tokens(rendered))
    
    def render_for_prompt(self, state: MemoryState) -> str:
        struct_state: StructuredState = state.state
        
        if struct_state.rounds_seen == 0:
            return "[No history yet - trust scores initialized to neutral (0.5)]"
        
        lines = [
            "=== Trust Table (based on cooperation history) ===",
            "",
            "| Agent      | Trust | Last | Avg  | Defections |",
            "|------------|-------|------|------|------------|",
        ]
        
        for agent_id, record in sorted(struct_state.records.items()):
            if agent_id == struct_state.agent_id:
                label = "You       "
            else:
                label = f"Agent {agent_id}   "
            
            lines.append(
                f"| {label} | {record.cooperation_score:.2f}  | {record.last_contribution:4d} | "
                f"{record.avg_contribution:.1f}  | {record.defection_count:10d} |"
            )
        
        lines.append("")
        lines.append(f"(Trust: 0=always defects, 1=always cooperates. Based on {struct_state.rounds_seen} rounds)")
        
        return "\n".join(lines)

