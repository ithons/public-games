"""
Sanity Tests for the LLM Agent Memory Project

Verifies:
1. Environment payoff formula is correct
2. Fixed contribution patterns produce deterministic outcomes
3. Repeated runs with same seed produce identical trajectories
4. Gini coefficient implementation is correct
5. Metrics definitions are as documented
"""

import pytest
import numpy as np
from environment.public_goods_env import PublicGoodsEnv, EnvConfig


class TestPayoffFormula:
    """Verify the payoff formula: payoff_i = (α * Σc_j) / N - c_i"""
    
    def test_all_contribute_equally(self):
        """When all agents contribute c, payoff should be (α-1)*c for each."""
        config = EnvConfig(
            num_agents=3,
            num_rounds=1,
            starting_budget=20.0,
            max_contribution=10,
            multiplier=1.8,
        )
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        contribution = 5
        result = env.step([contribution, contribution, contribution])
        
        # payoff = (1.8 * 15) / 3 - 5 = 9 - 5 = 4
        expected_payoff = (config.multiplier * contribution * 3) / 3 - contribution
        expected_payoff = (config.multiplier - 1) * contribution  # Simplified: (α-1)*c
        
        for i in range(3):
            assert abs(result.rewards[i] - expected_payoff) < 1e-10, (
                f"Expected payoff {expected_payoff}, got {result.rewards[i]}"
            )
    
    def test_all_max_contribution_welfare(self):
        """When everyone contributes max, welfare should be 3 * (α-1) * c_max per round."""
        config = EnvConfig(
            num_agents=3,
            num_rounds=10,
            starting_budget=100.0,  # Enough for 10 rounds of max
            max_contribution=10,
            multiplier=1.8,
        )
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        total_welfare = 0.0
        for _ in range(10):
            result = env.step([10, 10, 10])
            total_welfare += sum(result.rewards)
        
        # Each round: 3 agents * (1.8-1) * 10 = 24 welfare
        # 10 rounds: 240 total welfare
        expected = 3 * (config.multiplier - 1) * config.max_contribution * config.num_rounds
        assert abs(total_welfare - expected) < 1e-6, (
            f"Expected welfare {expected}, got {total_welfare}"
        )
    
    def test_free_rider_gets_more(self):
        """A free-rider should gain more than cooperators in a single round."""
        config = EnvConfig(num_agents=3, num_rounds=1, multiplier=1.8)
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        # Agent 0 defects, others cooperate maximally
        result = env.step([0, 10, 10])
        
        # Free-rider payoff: (1.8 * 20) / 3 - 0 = 12
        free_rider_payoff = (config.multiplier * 20) / 3 - 0
        # Cooperator payoff: (1.8 * 20) / 3 - 10 = 2
        cooperator_payoff = (config.multiplier * 20) / 3 - 10
        
        assert abs(result.rewards[0] - free_rider_payoff) < 1e-6
        assert abs(result.rewards[1] - cooperator_payoff) < 1e-6
        assert result.rewards[0] > result.rewards[1], "Free-rider should gain more"
    
    def test_budget_updates_correctly(self):
        """Budget should update as: budget - c + share."""
        config = EnvConfig(num_agents=3, num_rounds=1, starting_budget=20.0, multiplier=1.8)
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        # Agent contributes 6, others contribute 4 each
        result = env.step([6, 4, 4])
        
        # Pot = 14, multiplied = 14 * 1.8 = 25.2, share = 8.4
        pot = 14
        multiplied_pot = pot * config.multiplier
        share = multiplied_pot / 3
        
        # Agent 0: 20 - 6 + 8.4 = 22.4
        expected_budget_0 = 20 - 6 + share
        assert abs(result.observations[0].budget - expected_budget_0) < 1e-6


class TestReproducibility:
    """Verify that experiments are reproducible with the same seed."""
    
    def test_same_seed_same_initial_state(self):
        """Same seed should produce identical initial state."""
        env1 = PublicGoodsEnv()
        env2 = PublicGoodsEnv()
        
        obs1 = env1.reset(seed=12345)
        obs2 = env2.reset(seed=12345)
        
        for o1, o2 in zip(obs1, obs2):
            assert o1.budget == o2.budget
            assert o1.round_index == o2.round_index
    
    def test_deterministic_trajectory(self):
        """Fixed contributions should produce identical trajectory."""
        contributions_sequence = [
            [5, 5, 5],
            [7, 3, 5],
            [10, 0, 10],
        ]
        
        def run_trajectory(seed):
            config = EnvConfig(num_rounds=3)
            env = PublicGoodsEnv(config)
            env.reset(seed=seed)
            
            payoffs = []
            budgets = []
            for contribs in contributions_sequence:
                result = env.step(contribs)
                payoffs.append(result.rewards.copy())
                budgets.append([o.budget for o in result.observations])
            return payoffs, budgets
        
        payoffs1, budgets1 = run_trajectory(42)
        payoffs2, budgets2 = run_trajectory(42)
        
        assert payoffs1 == payoffs2, "Payoffs should be identical"
        assert budgets1 == budgets2, "Budgets should be identical"


class TestGiniCoefficient:
    """Verify Gini coefficient implementation."""
    
    def test_gini_perfect_equality(self):
        """All equal values should give Gini = 0."""
        from analysis.analyze_results import gini_coefficient
        
        assert abs(gini_coefficient([10, 10, 10])) < 1e-10
        assert abs(gini_coefficient([1, 1, 1, 1, 1])) < 1e-10
        assert abs(gini_coefficient([100.0])) < 1e-10  # Single value
    
    def test_gini_maximum_inequality(self):
        """One person has everything should give Gini close to 1."""
        from analysis.analyze_results import gini_coefficient
        
        # With N items where one has all, Gini = (N-1)/N
        result = gini_coefficient([100, 0, 0])
        expected = 2 / 3  # (3-1)/3
        assert abs(result - expected) < 0.01
    
    def test_gini_known_value(self):
        """Test against a known Gini value."""
        from analysis.analyze_results import gini_coefficient
        
        # For [1, 2, 3, 4, 5], Gini ≈ 0.2667
        result = gini_coefficient([1, 2, 3, 4, 5])
        assert 0.2 < result < 0.3


class TestWelfareBounds:
    """Verify welfare bounds match theory."""
    
    def test_theoretical_max_welfare(self):
        """Maximum welfare occurs when all contribute max."""
        config = EnvConfig(
            num_agents=3,
            num_rounds=10,
            starting_budget=100.0,
            max_contribution=10,
            multiplier=1.8,
        )
        
        # Theoretical max: N * (α-1) * c_max * T
        # = 3 * 0.8 * 10 * 10 = 240
        theoretical_max = (
            config.num_agents * 
            (config.multiplier - 1) * 
            config.max_contribution * 
            config.num_rounds
        )
        assert abs(theoretical_max - 240.0) < 1e-6
        
        # Actually run and verify
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        total_welfare = 0.0
        for _ in range(10):
            result = env.step([10, 10, 10])
            total_welfare += sum(result.rewards)
        
        assert abs(total_welfare - theoretical_max) < 1e-6
    
    def test_all_defect_welfare(self):
        """When all defect, welfare should be 0."""
        config = EnvConfig(num_agents=3, num_rounds=10, multiplier=1.8)
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        total_welfare = 0.0
        for _ in range(10):
            result = env.step([0, 0, 0])
            total_welfare += sum(result.rewards)
        
        assert total_welfare == 0.0


class TestCooperationMetrics:
    """Verify cooperation metrics are computed correctly."""
    
    def test_cooperation_rate_all_max(self):
        """When all contribute max, cooperation rate should be 1.0."""
        # threshold = 0.5 * c_max = 5, so contributions >= 5 count as cooperation
        contributions = [[10, 10, 10]] * 10  # 10 rounds, all max
        
        # Flatten and count
        all_c = [c for round_c in contributions for c in round_c]
        high_threshold = 5  # 0.5 * 10
        coop_rate = sum(1 for c in all_c if c >= high_threshold) / len(all_c)
        
        assert coop_rate == 1.0
    
    def test_cooperation_rate_all_zero(self):
        """When all contribute 0, cooperation rate should be 0.0."""
        contributions = [[0, 0, 0]] * 10
        all_c = [c for round_c in contributions for c in round_c]
        high_threshold = 5
        coop_rate = sum(1 for c in all_c if c >= high_threshold) / len(all_c)
        
        assert coop_rate == 0.0
    
    def test_cooperation_rate_mixed(self):
        """Test mixed contributions give expected rate."""
        # 2 agents at 6 (cooperative), 1 at 2 (not cooperative), for 10 rounds
        contributions = [[6, 6, 2]] * 10
        all_c = [c for round_c in contributions for c in round_c]
        high_threshold = 5
        coop_rate = sum(1 for c in all_c if c >= high_threshold) / len(all_c)
        
        # 20 cooperative out of 30 = 2/3
        assert abs(coop_rate - 2/3) < 1e-10


class TestStabilityMetric:
    """Verify stability metric implementation."""
    
    def test_stability_constant_contributions(self):
        """Constant contributions should give stability close to 1."""
        # If all round means are the same, variance = 0, stability = 1
        mean_per_round = [5.0] * 10  # Same mean every round
        variance = np.var(mean_per_round)
        c_max = 10
        var_max = (c_max / 2) ** 2  # Variance of alternating [0, 10, 0, 10, ...]
        stability = 1 - variance / var_max if var_max > 0 else 1.0
        
        assert abs(stability - 1.0) < 1e-10
    
    def test_stability_alternating_contributions(self):
        """Alternating [0, 10] should give low stability."""
        mean_per_round = [0, 10, 0, 10, 0, 10, 0, 10, 0, 10]
        variance = np.var(mean_per_round)  # Should be 25
        c_max = 10
        var_max = (c_max / 2) ** 2  # = 25
        stability = 1 - variance / var_max
        
        assert abs(stability - 0.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
