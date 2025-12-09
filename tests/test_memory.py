"""
Tests for Memory Modules.
"""

import pytest
from memory import create_memory, MEMORY_REGISTRY
from memory.base_memory import RoundInfo
from memory.no_memory import NoMemory
from memory.full_history import FullHistoryMemory
from memory.summary_memory import SummaryMemory
from memory.structured_memory import StructuredMemory
from memory.hybrid_memory import HybridMemory


def make_round_info(
    round_index: int = 0,
    agent_id: int = 0,
    contributions: list[int] = None,
    pot: float = 30.0,
    payoffs: list[float] = None,
) -> RoundInfo:
    """Helper to create RoundInfo for testing."""
    return RoundInfo(
        round_index=round_index,
        agent_id=agent_id,
        contributions=contributions or [5, 5, 5],
        pot=pot,
        payoffs=payoffs or [5.0, 5.0, 5.0],
        own_budget=15.0,
        own_cumulative_payoff=5.0,
    )


class TestMemoryRegistry:
    def test_all_memory_types_registered(self):
        expected = {"none", "full_history", "summary", "structured", "hybrid", "hybrid_neutral"}
        assert set(MEMORY_REGISTRY.keys()) == expected
    
    def test_create_memory_factory(self):
        for mem_type in MEMORY_REGISTRY:
            memory = create_memory(mem_type)
            assert memory is not None
    
    def test_create_memory_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown memory type"):
            create_memory("nonexistent")


class TestNoMemory:
    def test_init_state(self):
        memory = NoMemory()
        state = memory.init_state(agent_id=0, num_agents=3)
        
        assert state.state["agent_id"] == 0
        assert state.state["num_agents"] == 3
    
    def test_update_does_nothing(self):
        memory = NoMemory()
        state = memory.init_state(0, 3)
        
        round_info = make_round_info()
        new_state = memory.update(state, round_info)
        
        assert new_state.state == state.state
    
    def test_render_for_prompt(self):
        memory = NoMemory()
        state = memory.init_state(0, 3)
        
        rendered = memory.render_for_prompt(state)
        assert "no memory" in rendered.lower()


class TestFullHistoryMemory:
    def test_init_state(self):
        memory = FullHistoryMemory(window_size=5)
        state = memory.init_state(agent_id=1, num_agents=3)
        
        assert len(state.state.history) == 0
        assert state.state.window_size == 5
    
    def test_update_adds_entry(self):
        memory = FullHistoryMemory(window_size=5)
        state = memory.init_state(0, 3)
        
        round_info = make_round_info(round_index=0, contributions=[3, 5, 7])
        new_state = memory.update(state, round_info)
        
        assert len(new_state.state.history) == 1
        assert new_state.state.history[0].contributions == [3, 5, 7]
    
    def test_window_truncation(self):
        memory = FullHistoryMemory(window_size=2)
        state = memory.init_state(0, 3)
        
        for i in range(5):
            round_info = make_round_info(round_index=i, contributions=[i, i, i])
            state = memory.update(state, round_info)
        
        # Should only keep last 2 rounds
        assert len(state.state.history) == 2
        assert state.state.history[0].round_index == 3
        assert state.state.history[1].round_index == 4
    
    def test_render_includes_history(self):
        memory = FullHistoryMemory(window_size=5)
        state = memory.init_state(0, 3)
        
        state = memory.update(state, make_round_info(round_index=0, contributions=[5, 3, 7]))
        rendered = memory.render_for_prompt(state)
        
        assert "Round 1" in rendered
        assert "You: 5" in rendered


class TestSummaryMemory:
    def test_init_state(self):
        memory = SummaryMemory(max_words=30)
        state = memory.init_state(0, 3)
        
        assert "No history yet" in state.state.summary
        assert state.state.max_words == 30
    
    def test_update_with_llm_response(self):
        memory = SummaryMemory(max_words=30)
        state = memory.init_state(0, 3)
        
        round_info = make_round_info()
        llm_response = "Updated summary: All agents cooperated in round 1."
        
        new_state = memory.update(state, round_info, llm_response)
        
        assert "cooperated" in new_state.state.summary.lower()
    
    def test_render_includes_summary(self):
        memory = SummaryMemory(max_words=30)
        state = memory.init_state(0, 3)
        
        rendered = memory.render_for_prompt(state)
        assert "Summary" in rendered


class TestStructuredMemory:
    def test_init_state(self):
        memory = StructuredMemory()
        state = memory.init_state(agent_id=0, num_agents=3)
        
        assert len(state.state.records) == 3
        assert all(r.cooperation_score == 0.5 for r in state.state.records.values())
    
    def test_update_cooperation_score(self):
        memory = StructuredMemory()
        state = memory.init_state(0, 3)
        
        # High contribution should increase cooperation score
        round_info = make_round_info(contributions=[10, 10, 10])
        new_state = memory.update(state, round_info)
        
        # Score should have increased from 0.5 toward 1.0
        for record in new_state.state.records.values():
            assert record.cooperation_score > 0.5
    
    def test_update_defection_count(self):
        memory = StructuredMemory(fair_share_fraction=0.5)
        state = memory.init_state(0, 3)
        
        # Agent 1 defects (contributes 0 while others contribute 8)
        round_info = make_round_info(contributions=[8, 0, 8])
        new_state = memory.update(state, round_info)
        
        # Agent 1 should have a defection recorded
        assert new_state.state.records[1].defection_count >= 1
    
    def test_render_produces_table(self):
        memory = StructuredMemory()
        state = memory.init_state(0, 3)
        state = memory.update(state, make_round_info())
        
        rendered = memory.render_for_prompt(state)
        
        assert "Trust" in rendered
        assert "Agent" in rendered
        assert "|" in rendered  # Table format


class TestHybridMemory:
    def test_init_state(self):
        memory = HybridMemory(max_note_words=20)
        state = memory.init_state(0, 3)
        
        assert len(state.state.structured.records) == 3
        assert "cooperat" in state.state.strategy_note.lower()
    
    def test_update_both_components(self):
        memory = HybridMemory()
        state = memory.init_state(0, 3)
        
        round_info = make_round_info(contributions=[10, 10, 10])
        llm_response = "Strategy note: Keep cooperating since everyone is trustworthy."
        
        new_state = memory.update(state, round_info, llm_response)
        
        # Structured part should update
        for record in new_state.state.structured.records.values():
            assert record.cooperation_score > 0.5
        
        # Strategy note should update
        assert "cooperating" in new_state.state.strategy_note.lower()
    
    def test_render_includes_both(self):
        memory = HybridMemory()
        state = memory.init_state(0, 3)
        state = memory.update(state, make_round_info())
        
        rendered = memory.render_for_prompt(state)
        
        assert "Trust" in rendered  # Table
        assert "Strategy Note" in rendered  # Note section

