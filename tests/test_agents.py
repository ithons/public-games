"""
Tests for Agent Policies.
"""

import pytest
from agents.base_policy import RandomPolicy, FixedPolicy, TitForTatPolicy, AgentAction
from agents.llm_policy import LLMPolicy, MockLLMBackend, LLMResponse
from memory.no_memory import NoMemory
from memory.structured_memory import StructuredMemory
from environment.public_goods_env import RoundObservation


def make_observation(
    round_index: int = 0,
    agent_id: int = 0,
    budget: float = 20.0,
    cumulative_payoff: float = 0.0,
    last_contributions: list[int] = None,
) -> RoundObservation:
    """Helper to create observations for testing."""
    return RoundObservation(
        round_index=round_index,
        agent_id=agent_id,
        budget=budget,
        cumulative_payoff=cumulative_payoff,
        last_round_contributions=last_contributions,
        last_round_pot=None if last_contributions is None else sum(last_contributions) * 1.8,
        last_round_payoffs=None,
    )


class TestRandomPolicy:
    def test_act_within_bounds(self):
        policy = RandomPolicy(seed=42)
        memory = NoMemory()
        state = memory.init_state(0, 3)
        obs = make_observation(budget=15.0)
        
        for _ in range(10):
            action = policy.act(obs, state, max_contribution=10)
            assert 0 <= action.contribution <= 10
            assert action.contribution <= 15  # Within budget
    
    def test_reproducible_with_seed(self):
        policy1 = RandomPolicy(seed=42)
        policy2 = RandomPolicy(seed=42)
        memory = NoMemory()
        state = memory.init_state(0, 3)
        obs = make_observation()
        
        actions1 = [policy1.act(obs, state, 10).contribution for _ in range(5)]
        actions2 = [policy2.act(obs, state, 10).contribution for _ in range(5)]
        
        assert actions1 == actions2


class TestFixedPolicy:
    def test_fixed_contribution(self):
        policy = FixedPolicy(contribution=7)
        memory = NoMemory()
        state = memory.init_state(0, 3)
        obs = make_observation()
        
        action = policy.act(obs, state, max_contribution=10)
        assert action.contribution == 7
    
    def test_clips_to_max(self):
        policy = FixedPolicy(contribution=15)
        memory = NoMemory()
        state = memory.init_state(0, 3)
        obs = make_observation()
        
        action = policy.act(obs, state, max_contribution=10)
        assert action.contribution == 10
    
    def test_clips_to_budget(self):
        policy = FixedPolicy(contribution=10)
        memory = NoMemory()
        state = memory.init_state(0, 3)
        obs = make_observation(budget=5.0)
        
        action = policy.act(obs, state, max_contribution=10)
        assert action.contribution == 5


class TestTitForTatPolicy:
    def test_initial_cooperation(self):
        policy = TitForTatPolicy(initial_contribution=5)
        memory = NoMemory()
        state = memory.init_state(0, 3)
        obs = make_observation(round_index=0)
        
        action = policy.act(obs, state, max_contribution=10)
        assert action.contribution == 5
    
    def test_matches_others_average(self):
        policy = TitForTatPolicy()
        policy.reset()
        memory = NoMemory()
        state = memory.init_state(0, 3)
        
        # First round
        obs1 = make_observation(round_index=0)
        policy.act(obs1, state, 10)
        
        # Second round with known contributions
        obs2 = make_observation(round_index=1, agent_id=0, last_contributions=[5, 8, 2])
        action = policy.act(obs2, state, 10)
        
        # Should match average of others (8 + 2) / 2 = 5
        assert action.contribution == 5


class TestMockLLMBackend:
    def test_cooperative_strategy(self):
        backend = MockLLMBackend(strategy="cooperative")
        
        prompt = "Round 1. Budget: 20. Maximum: 10"
        response = backend.complete(prompt)
        
        assert response.content is not None
        assert "contribution" in response.content.lower()
    
    def test_defector_strategy(self):
        backend = MockLLMBackend(strategy="defector")
        
        prompt = "Round 1. Budget: 20. Maximum: 10"
        response = backend.complete(prompt)
        
        # Should produce valid response
        assert "json" in response.content.lower()
    
    def test_token_estimation(self):
        backend = MockLLMBackend()
        
        prompt = "This is a test prompt with about 40 characters."
        response = backend.complete(prompt)
        
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0


class TestLLMPolicy:
    def test_act_returns_valid_action(self):
        backend = MockLLMBackend(strategy="cooperative")
        memory = NoMemory()
        policy = LLMPolicy(backend=backend, memory=memory)
        
        state = memory.init_state(0, 3)
        obs = make_observation(budget=20.0)
        
        action = policy.act(obs, state, max_contribution=10)
        
        assert isinstance(action, AgentAction)
        assert 0 <= action.contribution <= 10
    
    def test_clips_contribution_to_budget(self):
        backend = MockLLMBackend(strategy="cooperative")
        memory = NoMemory()
        policy = LLMPolicy(backend=backend, memory=memory)
        
        state = memory.init_state(0, 3)
        obs = make_observation(budget=3.0)
        
        action = policy.act(obs, state, max_contribution=10)
        
        assert action.contribution <= 3
    
    def test_tracks_tokens(self):
        backend = MockLLMBackend()
        memory = NoMemory()
        policy = LLMPolicy(backend=backend, memory=memory)
        
        state = memory.init_state(0, 3)
        obs = make_observation()
        
        policy.act(obs, state, 10)
        
        prompt_tokens, completion_tokens = policy.total_tokens
        assert prompt_tokens > 0
        assert completion_tokens > 0
    
    def test_reset_clears_tokens(self):
        backend = MockLLMBackend()
        memory = NoMemory()
        policy = LLMPolicy(backend=backend, memory=memory)
        
        state = memory.init_state(0, 3)
        obs = make_observation()
        
        policy.act(obs, state, 10)
        policy.reset()
        
        prompt_tokens, completion_tokens = policy.total_tokens
        assert prompt_tokens == 0
        assert completion_tokens == 0
    
    def test_with_structured_memory(self):
        backend = MockLLMBackend()
        memory = StructuredMemory()
        policy = LLMPolicy(backend=backend, memory=memory)
        
        state = memory.init_state(0, 3)
        obs = make_observation(budget=20.0, last_contributions=[5, 5, 5])
        
        action = policy.act(obs, state, max_contribution=10)
        
        assert isinstance(action, AgentAction)
        assert action.contribution >= 0

