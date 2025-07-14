# models.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import networkx as nx


@dataclass
class SubTask:
    id: str
    description: str
    mode: str | None = None     # "WIDE" or "DEEP"


@dataclass
class Workflow:
    task_id: str
    original_prompt: str
    subtasks: List[SubTask] = field(default_factory=list)
    dag: nx.DiGraph = field(default_factory=nx.DiGraph)
    complexity: float | None = None
    report: Dict | None = None


@dataclass
class ModeResult:
    primary_mode: str
    confidence: float
    is_hybrid: bool
    secondary_mode: str | None = None
    execution_strategy: str | None = None
    phase_breakdown: List[Dict] | None = None
    information_requirements: List[str] | None = None
    processing_complexity: str = "medium"


@dataclass
class ComplexityMetrics:
    base_score: float
    uncertainty_score: float
    domain_complexity: int
    coordination_complexity: int
    computational_complexity: int
    temporal_uncertainty: int
    resource_uncertainty: int
    outcome_uncertainty: int
    critical_path_factor: float
    parallel_efficiency: float
    mode_heterogeneity: float
    resource_conflict_factor: float
    requires_replanning: bool
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass 
class WorkflowQualityMetrics:
    completeness_score: float  # 0-1: How well subtasks cover the main task
    coherence_score: float     # 0-1: How logically connected the workflow is
    efficiency_score: float    # 0-1: How well structured for parallel execution
    feasibility_score: float   # 0-1: How realistic the workflow is
    overall_quality: float     # 0-1: Combined quality score
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ExecutionEstimate:
    estimated_total_time: float  # in hours
    critical_path_time: float
    parallel_time_savings: float
    resource_requirements: Dict[str, int]
    cost_estimate: Dict[str, float]
    bottlenecks: List[str] = field(default_factory=list)