"""
Tests for the Public Goods Environment.
"""

import pytest
from environment.public_goods_env import PublicGoodsEnv, EnvConfig, compute_nash_equilibrium


class TestEnvConfig:
    def test_default_config(self):
        config = EnvConfig()
        assert config.num_agents == 3
        assert config.num_rounds == 10
        assert config.starting_budget == 20.0
        assert config.max_contribution == 10
        assert 1 < config.multiplier < config.num_agents
    
    def test_invalid_multiplier_too_low(self):
        with pytest.raises(ValueError, match="must be > 1"):
            EnvConfig(multiplier=0.5)
    
    def test_invalid_multiplier_too_high(self):
        with pytest.raises(ValueError, match="must be < num_agents"):
            EnvConfig(num_agents=3, multiplier=4.0)


class TestPublicGoodsEnv:
    def test_reset_initializes_state(self):
        env = PublicGoodsEnv()
        obs = env.reset(seed=42)
        
        assert len(obs) == 3
        assert all(o.round_index == 0 for o in obs)
        assert all(o.budget == 20.0 for o in obs)
        assert all(o.cumulative_payoff == 0.0 for o in obs)
        assert all(o.last_round_contributions is None for o in obs)
    
    def test_reset_is_reproducible(self):
        env = PublicGoodsEnv()
        
        obs1 = env.reset(seed=42)
        obs2 = env.reset(seed=42)
        
        assert obs1[0].budget == obs2[0].budget
    
    def test_step_validates_contribution_count(self):
        env = PublicGoodsEnv()
        env.reset(seed=42)
        
        with pytest.raises(ValueError, match="Expected 3"):
            env.step([5, 5])  # Only 2 contributions
    
    def test_step_clips_contributions(self):
        env = PublicGoodsEnv()
        env.reset(seed=42)
        
        # Contribute more than max - should be clipped
        result = env.step([100, 5, 5])
        
        # Max is 10, so first contribution should be clipped
        assert result.info["contributions"][0] == 10
    
    def test_step_clips_to_budget(self):
        config = EnvConfig(starting_budget=5.0, max_contribution=10)
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        # Contribute more than budget
        result = env.step([10, 10, 10])
        
        # Should be clipped to budget (5)
        assert all(c == 5 for c in result.info["contributions"])
    
    def test_payoff_calculation(self):
        config = EnvConfig(num_agents=3, multiplier=1.5)
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        # All contribute 6
        result = env.step([6, 6, 6])
        
        pot = 6 * 3 * 1.5  # 27
        share = pot / 3  # 9
        expected_reward = share - 6  # 3
        
        assert abs(result.rewards[0] - expected_reward) < 0.001
        assert result.info["pot"] == pot
    
    def test_free_rider_advantage(self):
        config = EnvConfig(num_agents=3, multiplier=1.5)
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        # Agent 0 defects, others cooperate
        result = env.step([0, 6, 6])
        
        pot = 12 * 1.5  # 18
        share = 6  # pot / 3
        
        # Defector gets share without contributing
        assert result.rewards[0] == share - 0  # 6
        # Cooperators get share minus contribution
        assert result.rewards[1] == share - 6  # 0
    
    def test_done_after_all_rounds(self):
        config = EnvConfig(num_rounds=3)
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        result1 = env.step([5, 5, 5])
        assert not result1.done
        
        result2 = env.step([5, 5, 5])
        assert not result2.done
        
        result3 = env.step([5, 5, 5])
        assert result3.done
    
    def test_episode_logging(self):
        config = EnvConfig(num_rounds=2)
        env = PublicGoodsEnv(config)
        env.reset(seed=42)
        
        env.step([3, 4, 5])
        env.step([5, 5, 5])
        
        log = env.get_episode_log()
        assert log is not None
        assert len(log.rounds) == 2
        assert log.rounds[0]["contributions"] == [3, 4, 5]


class TestNashEquilibrium:
    def test_social_dilemma_detection(self):
        # Standard social dilemma: α/n < 1 < α
        config = EnvConfig(num_agents=3, multiplier=1.8)
        result = compute_nash_equilibrium(config)
        
        assert result["is_social_dilemma"]
        assert result["marginal_return_per_unit"] == 0.6  # 1.8/3
        assert result["nash_contribution"] == 0
        assert result["social_optimal_contribution"] == config.max_contribution

