"""
Test Blog Consistency

Ensures that numeric claims in the blog match actual experiment results.
"""

import pytest
from pathlib import Path

from tools.check_blog_consistency import (
    run_consistency_check,
    load_summary_metrics,
    load_ablation_metrics,
    load_alpha_sweep_metrics,
)


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


class TestBlogConsistency:
    """Test that blog claims match actual data."""
    
    def test_consistency_check_passes(self, project_root):
        """The main consistency check should find no mismatches."""
        results_dir = project_root / "results"
        blog_path = project_root / "blog" / "index.html"
        
        if not results_dir.exists():
            pytest.skip("Results directory not found")
        if not blog_path.exists():
            pytest.skip("Blog file not found")
        
        mismatches = run_consistency_check(results_dir, blog_path)
        
        if mismatches:
            mismatch_strs = "\n".join(str(m) for m in mismatches)
            pytest.fail(f"Found {len(mismatches)} mismatch(es):\n{mismatch_strs}")
    
    def test_summary_metrics_exist(self, project_root):
        """Summary metrics file should exist."""
        results_dir = project_root / "results"
        
        if not results_dir.exists():
            pytest.skip("Results directory not found")
        
        metrics = load_summary_metrics(results_dir)
        
        # Check all expected conditions exist
        expected_conditions = ["none", "full_history", "summary", "structured", "hybrid"]
        for condition in expected_conditions:
            assert condition in metrics, f"Missing condition: {condition}"
    
    def test_main_welfare_values(self, project_root):
        """Check that main welfare values are correct."""
        results_dir = project_root / "results"
        
        if not results_dir.exists():
            pytest.skip("Results directory not found")
        
        metrics = load_summary_metrics(results_dir)
        
        # Check hybrid achieves theoretical maximum (240)
        assert abs(metrics["hybrid"]["welfare"]["mean"] - 240.0) < 0.1
        
        # Check non-hybrid conditions are around 120
        for condition in ["none", "full_history"]:
            assert 118 < metrics[condition]["welfare"]["mean"] < 125
    
    def test_ablation_metrics(self, project_root):
        """Check that ablation metrics exist and make sense."""
        results_dir = project_root / "results"
        
        if not results_dir.exists():
            pytest.skip("Results directory not found")
        
        ablation = load_ablation_metrics(results_dir)
        
        # Should have none, structured, and hybrid
        assert "none" in ablation or len(ablation) == 0  # Skip if no ablation data
        
        if "hybrid" in ablation:
            # Hybrid should achieve much higher welfare
            assert ablation["hybrid"]["welfare_mean"] > ablation.get("none", {}).get("welfare_mean", 0) * 1.5
    
    def test_alpha_sweep_metrics(self, project_root):
        """Check that alpha sweep metrics exist."""
        results_dir = project_root / "results"
        
        if not results_dir.exists():
            pytest.skip("Results directory not found")
        
        sweep = load_alpha_sweep_metrics(results_dir)
        
        # Should have multiple alpha values if sweep was run
        if sweep:
            alphas = set(m["alpha"] for m in sweep.values())
            # Should have at least 2 different alpha values
            assert len(alphas) >= 2 or len(sweep) == 0
