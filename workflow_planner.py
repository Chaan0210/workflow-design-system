# workflow_planner.py
from typing import Tuple, Dict
import networkx as nx

from models import Workflow, SubTask, ComplexityMetrics, WorkflowQualityMetrics, ExecutionEstimate
from decomposition import decompose
from dag_builder import build_dag
from mode_classifier import classify_mode, determine_optimal_mode_sequence
from complexity_analyzer import compute_complexity, should_replan
from validator import validate_workflow_quality, estimate_execution_metrics, diagnose_workflow_issues


# ------------------------ Stage 5: Full pipeline -------------------------- #

def plan_workflow(task_id: str, prompt: str) -> Tuple[Workflow, ComplexityMetrics, WorkflowQualityMetrics, ExecutionEstimate, Dict]:
    wf = Workflow(task_id=task_id, original_prompt=prompt)
    
    print(f"🔍 Planning workflow for task: {task_id}")
    
    # 1) Task decomposition
    print("  📋 Decomposing into sub-tasks...")
    wf.subtasks = decompose(prompt)
    print(f"     Generated {len(wf.subtasks)} sub-tasks")
    
    # 2) DAG construction
    print("  🔗 Building dependency graph...")
    wf.dag = build_dag(wf.subtasks)
    print(f"     Created DAG with {wf.dag.number_of_edges()} dependencies")
    
    # 3) WIDE/DEEP classification
    print("  🎯 Analyzing task modes...")
    mode_results = {}
    for st in wf.subtasks:
        mode_result = classify_mode(st, prompt)
        mode_results[st.id] = mode_result
        st.mode = mode_result.primary_mode
        wf.dag.nodes[st.id]["obj"].mode = st.mode
    
    wide_count = len([st for st in wf.subtasks if st.mode == "WIDE"])
    deep_count = len([st for st in wf.subtasks if st.mode == "DEEP"])
    print(f"     Classified: {wide_count} WIDE, {deep_count} DEEP tasks")
    
    # 4) Complexity analysis
    print("  📊 Computing complexity metrics...")
    complexity_metrics = compute_complexity(wf)
    wf.complexity = complexity_metrics.base_score
    print(f"     Complexity: {complexity_metrics.base_score:.2f}, Uncertainty: {complexity_metrics.uncertainty_score:.3f}")
    
    # 5) Quality validation
    print("  ✅ Validating workflow quality...")
    quality_metrics = validate_workflow_quality(wf)
    print(f"     Quality score: {quality_metrics.overall_quality:.3f}/1.0")
    
    # 6) Execution estimation
    print("  ⏱️ Estimating execution metrics...")
    execution_estimate = estimate_execution_metrics(wf)
    print(f"     Estimated time: {execution_estimate.estimated_total_time:.1f}h (saves {execution_estimate.parallel_time_savings:.1f}h)")
    
    # 7) Issue diagnosis
    print("  🔍 Diagnosing potential issues...")
    issues = diagnose_workflow_issues(wf, quality_metrics, complexity_metrics)
    
    critical_count = len(issues.get('critical_issues', []))
    if critical_count > 0:
        print(f"     ⚠️ Found {critical_count} critical issues")
    
    # 8) Re-planning check
    if should_replan(complexity_metrics):
        print("     🔄 Re-planning recommended due to high complexity/uncertainty")
        issues['recommendations'].insert(0, "IMMEDIATE RE-PLANNING RECOMMENDED")
    
    # 9) Report generation
    parallel_blocks = {}
    for node in wf.dag.nodes():
        level = wf.dag.nodes[node].get('parallel_level', 0)
        if level not in parallel_blocks:
            parallel_blocks[level] = []
        parallel_blocks[level].append(node)
    
    # Get execution plan
    execution_plan = determine_optimal_mode_sequence(wf.subtasks, wf.dag, prompt)
    
    wf.report = {
        "task_id": wf.task_id,
        "complexity_score": complexity_metrics.base_score,
        "uncertainty_score": complexity_metrics.uncertainty_score,
        "quality_score": quality_metrics.overall_quality,
        "estimated_time_hours": execution_estimate.estimated_total_time,
        "parallel_time_savings": execution_estimate.parallel_time_savings,
        
        # Basic info
        "subtasks": [{"id": s.id, "desc": s.description, "mode": s.mode} for s in wf.subtasks],
        "dependencies": list(wf.dag.edges()),
        
        # Analysis
        "parallel_blocks": parallel_blocks,
        "critical_path": nx.algorithms.dag.longest_path(wf.dag) if wf.dag.nodes() else [],
        "execution_plan": execution_plan,
        
        # Metrics
        "complexity_metrics": {
            "base_score": complexity_metrics.base_score,
            "uncertainty_score": complexity_metrics.uncertainty_score,
            "domain_complexity": complexity_metrics.domain_complexity,
            "coordination_complexity": complexity_metrics.coordination_complexity,
            "parallel_efficiency": complexity_metrics.parallel_efficiency,
            "mode_heterogeneity": complexity_metrics.mode_heterogeneity,
            "requires_replanning": complexity_metrics.requires_replanning,
            "risk_factors": complexity_metrics.risk_factors,
            "mitigation_strategies": complexity_metrics.mitigation_strategies
        },
        
        "quality_metrics": {
            "completeness": quality_metrics.completeness_score,
            "coherence": quality_metrics.coherence_score,
            "efficiency": quality_metrics.efficiency_score,
            "feasibility": quality_metrics.feasibility_score,
            "overall_quality": quality_metrics.overall_quality,
            "validation_errors": quality_metrics.validation_errors,
            "warnings": quality_metrics.warnings,
            "suggestions": quality_metrics.suggestions
        },
        
        "execution_metrics": {
            "estimated_total_time": execution_estimate.estimated_total_time,
            "critical_path_time": execution_estimate.critical_path_time,
            "parallel_savings": execution_estimate.parallel_time_savings,
            "resource_requirements": execution_estimate.resource_requirements,
            "cost_estimate": execution_estimate.cost_estimate,
            "bottlenecks": execution_estimate.bottlenecks
        },
        
        "issues": issues
    }
    
    return wf, complexity_metrics, quality_metrics, execution_estimate, issues