"""End-to-end pipeline validation against synthetic data with known truth."""

from src.validation.pipeline import (
    PipelineOutput,
    run_pipeline,
    scenario_comparison,
)

__all__ = ["PipelineOutput", "run_pipeline", "scenario_comparison"]
