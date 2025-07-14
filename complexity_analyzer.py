# complexity_analyzer.py
import json
import math
from typing import List
import networkx as nx

from models import Workflow, SubTask, ComplexityMetrics
from utils import gpt


# -------------- Stage 4: Complexity / uncertainty metric -------------- #

BASE_W_SUBTASKS = 0.2
BASE_W_DEPENDENCIES = 0.3
BASE_W_AMBIGUITY = 0.25
BASE_W_CREATIVITY = 0.25

W_CRITICAL_PATH = 0.15
W_PARALLEL_EFFICIENCY = 0.1
W_MODE_HETEROGENEITY = 0.1
W_RESOURCE_CONFLICTS = 0.1

COMPLEXITY_ANALYSIS_TEMPLATE = """
Analyze the complexity and uncertainty of this task:

Task: "{task}"
Sub-tasks: {subtasks}

Rate each factor from 1-10:
1. Domain complexity (specialized knowledge required)
2. Coordination complexity (inter-task dependencies)  
3. Computational complexity (processing requirements)
4. Temporal uncertainty (unpredictable timing)
5. Resource uncertainty (variable resource needs)
6. Outcome uncertainty (unpredictable results)

Return JSON with:
{{
    "domain_complexity": 1-10,
    "coordination_complexity": 1-10, 
    "computational_complexity": 1-10,
    "temporal_uncertainty": 1-10,
    "resource_uncertainty": 1-10,
    "outcome_uncertainty": 1-10,
    "overall_uncertainty": 0.0-1.0,
    "requires_replanning": true/false,
    "risk_factors": ["factor1", "factor2", ...],
    "mitigation_strategies": ["strategy1", "strategy2", ...],
}}
"""


def ambiguity_score(text: str) -> float:
    fuzzy_words = ["etc", "unknown", "uncertain", "estimate", "maybe", "possibly", "might", 
                   "could", "approximately", "roughly", "probably", "perhaps", "somehow", 
                   "somewhat", "unclear", "vague", "ambiguous", "indefinite"]
    
    ambiguous_phrases = ["or something like that", "and so on", "among others", 
                        "to some extent", "kind of", "sort of", "more or less", 
                        "in some way", "to a degree", "somewhat like"]
    
    vague_quantifiers = ["some", "many", "few", "several", "various", "multiple",
                        "numerous", "certain", "quite a few", "a number of", "plenty of", 
                        "a bunch of", "loads of"]
    
    score = 0
    text_lower = text.lower()
    
    # Word-level ambiguity
    for word in fuzzy_words:
        score += text_lower.count(word) * 1.0
    
    # Phrase-level ambiguity  
    for phrase in ambiguous_phrases:
        score += text_lower.count(phrase) * 1.5
    
    # Quantifier ambiguity
    for quant in vague_quantifiers:
        score += text_lower.count(quant) * 0.7
    
    # Question marks indicate uncertainty
    score += text.count('?') * 0.5
    
    return score


def creativity_score(text: str) -> float:
    creative_keywords = ["innovate", "creative", "novel", "design", "invention", "breakthrough", 
                        "original", "unique", "groundbreaking", "innovative", "pioneering", 
                        "revolutionary", "cutting-edge", "state-of-the-art", "advanced"]
    
    creative_verbs = ["create", "invent", "design", "develop", "generate", "conceive",
                     "devise", "formulate", "construct", "build", "innovate", "pioneer", 
                     "establish", "originate", "craft", "fabricate", "manufacture"]
    
    creative_domains = ["art", "music", "literature", "architecture", "engineering",
                       "software", "algorithm", "solution", "approach", "method"]
    
    score = 0
    text_lower = text.lower()
    
    for word in creative_keywords:
        score += text_lower.count(word) * 1.5
    
    for verb in creative_verbs:
        score += text_lower.count(verb) * 1.2
    
    for domain in creative_domains:
        score += text_lower.count(domain) * 0.8
    
    return score


def calculate_critical_path_factor(dag: nx.DiGraph) -> float:
    if dag.number_of_nodes() == 0:
        return 0.0
    
    try:
        critical_path = nx.algorithms.dag.longest_path(dag)
        total_nodes = dag.number_of_nodes()
        critical_path_ratio = len(critical_path) / total_nodes
        
        return critical_path_ratio * 2.0
    except:
        return 1.0


def calculate_parallel_efficiency(dag: nx.DiGraph) -> float:
    if dag.number_of_nodes() == 0:
        return 1.0
    
    levels = {}
    for node in nx.topological_sort(dag):
        predecessors = list(dag.predecessors(node))
        if not predecessors:
            levels[node] = 0
        else:
            levels[node] = max(levels[pred] for pred in predecessors) + 1
    
    if not levels:
        return 1.0
    
    max_level = max(levels.values()) + 1
    total_nodes = len(levels)
    
    # Calculate average parallelism per level
    level_counts = {}
    for level in levels.values():
        level_counts[level] = level_counts.get(level, 0) + 1
    
    avg_parallelism = sum(level_counts.values()) / len(level_counts)
    theoretical_max_parallelism = total_nodes / max_level
    
    efficiency = avg_parallelism / theoretical_max_parallelism if theoretical_max_parallelism > 0 else 1.0
    return min(efficiency, 1.0)


def calculate_mode_heterogeneity(subtasks: List[SubTask]) -> float:
    if not subtasks:
        return 0.0
    
    modes = [st.mode for st in subtasks if st.mode]
    wide_count = modes.count("WIDE")
    deep_count = modes.count("DEEP")
    total = len(modes)
    
    if total == 0:
        return 0.0
    
    wide_ratio = wide_count / total
    deep_ratio = deep_count / total
    
    if wide_ratio > 0 and deep_ratio > 0:
        heterogeneity = -(wide_ratio * math.log2(wide_ratio) + deep_ratio * math.log2(deep_ratio))
        return heterogeneity  # Max value is 1.0 when evenly split
    else:
        return 0.0  # Homogeneous


def calculate_resource_conflict_factor(dag: nx.DiGraph) -> float:
    conflict_count = 0
    total_edges = dag.number_of_edges()
    
    for edge in dag.edges(data=True):
        if edge[2].get('has_resource_conflict', False):
            conflict_count += 1
    
    return (conflict_count / total_edges) if total_edges > 0 else 0.0


def compute_complexity(workflow: Workflow) -> ComplexityMetrics:
    # Basic metrics
    n_sub = len(workflow.subtasks)
    n_edges = workflow.dag.number_of_edges()
    
    # Scores
    amb_score = ambiguity_score(workflow.original_prompt)
    cre_score = creativity_score(workflow.original_prompt)
    
    # Advanced factors
    critical_path_factor = calculate_critical_path_factor(workflow.dag)
    parallel_efficiency = calculate_parallel_efficiency(workflow.dag)
    mode_heterogeneity = calculate_mode_heterogeneity(workflow.subtasks)
    resource_conflict_factor = calculate_resource_conflict_factor(workflow.dag)
    
    # Adaptive weights based on task characteristics
    weights = {
        'subtasks': BASE_W_SUBTASKS,
        'dependencies': BASE_W_DEPENDENCIES,
        'ambiguity': BASE_W_AMBIGUITY,
        'creativity': BASE_W_CREATIVITY,
        'critical_path': W_CRITICAL_PATH,
        'parallel_efficiency': W_PARALLEL_EFFICIENCY,
        'mode_heterogeneity': W_MODE_HETEROGENEITY,
        'resource_conflicts': W_RESOURCE_CONFLICTS
    }
    
    # Adjust weights based on task type
    if cre_score > 3:  # High creativity task
        weights['creativity'] *= 1.5
        weights['ambiguity'] *= 1.3
    
    if n_sub > 8:  # Large workflow
        weights['dependencies'] *= 1.4
        weights['critical_path'] *= 1.2
    
    # Calculate base complexity score
    base_score = (
        weights['subtasks'] * n_sub +
        weights['dependencies'] * n_edges +
        weights['ambiguity'] * amb_score +
        weights['creativity'] * cre_score +
        weights['critical_path'] * critical_path_factor +
        weights['parallel_efficiency'] * (2.0 - parallel_efficiency) +  # Lower efficiency = higher complexity
        weights['mode_heterogeneity'] * mode_heterogeneity +
        weights['resource_conflicts'] * resource_conflict_factor
    )
    
    # Get LLM-based uncertainty analysis
    try:
        subtask_desc = [{"id": st.id, "desc": st.description} for st in workflow.subtasks]
        uncertainty_analysis = json.loads(gpt(
            COMPLEXITY_ANALYSIS_TEMPLATE.format(
                task=workflow.original_prompt,
                subtasks=json.dumps(subtask_desc, ensure_ascii=False)
            ),
            system="Return valid JSON only.",
            temperature=0.1
        ))
        
        domain_complexity = uncertainty_analysis.get("domain_complexity", 5)
        coordination_complexity = uncertainty_analysis.get("coordination_complexity", 5)
        computational_complexity = uncertainty_analysis.get("computational_complexity", 5)
        temporal_uncertainty = uncertainty_analysis.get("temporal_uncertainty", 5)
        resource_uncertainty = uncertainty_analysis.get("resource_uncertainty", 5)
        outcome_uncertainty = uncertainty_analysis.get("outcome_uncertainty", 5)
        overall_uncertainty = uncertainty_analysis.get("overall_uncertainty", 0.5)
        requires_replanning = uncertainty_analysis.get("requires_replanning", False)
        risk_factors = uncertainty_analysis.get("risk_factors", [])
        mitigation_strategies = uncertainty_analysis.get("mitigation_strategies", [])
        
    except:
        # Fallback values
        domain_complexity = min(5 + int(cre_score), 10)
        coordination_complexity = min(3 + n_edges, 10)
        computational_complexity = min(2 + n_sub, 10)
        temporal_uncertainty = min(3 + int(amb_score), 10)
        resource_uncertainty = min(2 + int(resource_conflict_factor * 5), 10)
        outcome_uncertainty = min(4 + int(amb_score + cre_score), 10)
        overall_uncertainty = min((temporal_uncertainty + resource_uncertainty + outcome_uncertainty) / 30.0, 1.0)
        requires_replanning = base_score > 15 or overall_uncertainty > 0.8
        risk_factors = []
        mitigation_strategies = []
    
    # Combine base score with uncertainty
    uncertainty_multiplier = 1.0 + overall_uncertainty
    final_score = base_score * uncertainty_multiplier
    
    return ComplexityMetrics(
        base_score=round(base_score, 2),
        uncertainty_score=round(overall_uncertainty, 3),
        domain_complexity=domain_complexity,
        coordination_complexity=coordination_complexity,
        computational_complexity=computational_complexity,
        temporal_uncertainty=temporal_uncertainty,
        resource_uncertainty=resource_uncertainty,
        outcome_uncertainty=outcome_uncertainty,
        critical_path_factor=round(critical_path_factor, 3),
        parallel_efficiency=round(parallel_efficiency, 3),
        mode_heterogeneity=round(mode_heterogeneity, 3),
        resource_conflict_factor=round(resource_conflict_factor, 3),
        requires_replanning=requires_replanning,
        risk_factors=risk_factors,
        mitigation_strategies=mitigation_strategies
    )


def should_replan(complexity_metrics: ComplexityMetrics, threshold_score: float = 20.0, threshold_uncertainty: float = 0.8) -> bool:
    replan_conditions = [
        complexity_metrics.base_score > threshold_score,
        complexity_metrics.uncertainty_score > threshold_uncertainty,
        complexity_metrics.requires_replanning,
        complexity_metrics.outcome_uncertainty > 8,
        len(complexity_metrics.risk_factors) > 5
    ]
    
    return any(replan_conditions)
