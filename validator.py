# validator.py
import json
from typing import List, Dict
import networkx as nx

from models import Workflow, WorkflowQualityMetrics, ExecutionEstimate, ComplexityMetrics
from complexity_analyzer import calculate_parallel_efficiency
from utils import gpt


# ----------------------- Stage 4.5: Validation & Performance Metrics ----------------------- #

QUALITY_VALIDATION_TEMPLATE = """
Evaluate the quality of this workflow design:

Main Task: "{main_task}"
Sub-tasks: {subtasks}
Dependencies: {dependencies}

Rate each aspect from 0.0 to 1.0:
1. Completeness: Do the sub-tasks fully cover the main task?
2. Coherence: Are the dependencies logical and well-structured?
3. Efficiency: Is the workflow well-organized for execution?
4. Feasibility: Are all sub-tasks realistic and achievable?

Return JSON with:
{{
    "completeness_score": 0.0-1.0,
    "coherence_score": 0.0-1.0,
    "efficiency_score": 0.0-1.0,
    "feasibility_score": 0.0-1.0,
    "validation_errors": ["error1", "error2", ...],
    "warnings": ["warning1", "warning2", ...],
    "suggestions": ["suggestion1", "suggestion2", ...],
    "missing_subtasks": ["missing1", "missing2", ...],
    "redundant_subtasks": ["redundant1", "redundant2", ...],
    "problematic_dependencies": ["dep1", "dep2", ...]
}}
"""

EXECUTION_ESTIMATION_TEMPLATE = """
Estimate execution time and resource requirements for this workflow:

Sub-tasks with modes: {subtasks_with_modes}
Dependencies: {dependencies}
Parallel blocks: {parallel_blocks}

For each sub-task, estimate:
- Time required (in hours)
- CPU/compute requirements (low/medium/high)
- Memory requirements (low/medium/high)  
- Network/IO requirements (low/medium/high)

Return JSON with:
{{
    "task_estimates": {{
        "task_id": {{"time_hours": X, "cpu": "low/medium/high", "memory": "low/medium/high", "io": "low/medium/high"}},
        ...
    }},
    "critical_path_time": X.X,
    "total_sequential_time": X.X,
    "estimated_parallel_time": X.X,
    "bottlenecks": ["bottleneck1", "bottleneck2", ...],
    "resource_conflicts": ["conflict1", "conflict2", ...],
    "cost_factors": {{"compute": X.X, "storage": X.X, "network": X.X}}
}}
"""


def validate_workflow_quality(workflow: Workflow) -> WorkflowQualityMetrics:
    subtasks_info = [{"id": st.id, "desc": st.description, "mode": st.mode} for st in workflow.subtasks]
    dependencies_info = list(workflow.dag.edges())
    
    try:
        quality_analysis = json.loads(gpt(
            QUALITY_VALIDATION_TEMPLATE.format(
                main_task=workflow.original_prompt,
                subtasks=json.dumps(subtasks_info, ensure_ascii=False),
                dependencies=dependencies_info
            ),
            system="Return valid JSON only.",
            temperature=0.1
        ))
        
        completeness = quality_analysis.get("completeness_score", 0.7)
        coherence = quality_analysis.get("coherence_score", 0.7)
        efficiency = quality_analysis.get("efficiency_score", 0.7)
        feasibility = quality_analysis.get("feasibility_score", 0.7)
        
        validation_errors = quality_analysis.get("validation_errors", [])
        warnings = quality_analysis.get("warnings", [])
        suggestions = quality_analysis.get("suggestions", [])
        
    except Exception as e:
        # Fallback validation
        completeness = min(1.0, len(workflow.subtasks) / 5.0)  # Assume 5 subtasks = complete
        coherence = min(1.0, workflow.dag.number_of_edges() / len(workflow.subtasks)) if workflow.subtasks else 0.5
        efficiency = calculate_parallel_efficiency(workflow.dag)
        feasibility = 0.8  # Default assumption
        
        validation_errors = [f"LLM validation failed: {str(e)}"]
        warnings = []
        suggestions = ["Consider manual review due to validation failure"]
    
    # Additional programmatic checks
    if not workflow.subtasks:
        validation_errors.append("No sub-tasks generated")
        completeness = 0.0
    
    if workflow.dag.number_of_nodes() != len(workflow.subtasks):
        validation_errors.append("DAG nodes don't match sub-tasks")
        coherence *= 0.5
    
    if not nx.is_directed_acyclic_graph(workflow.dag):
        validation_errors.append("Workflow contains cycles")
        coherence = 0.0
    
    # Check for isolated nodes
    isolated_nodes = list(nx.isolates(workflow.dag))
    if isolated_nodes:
        warnings.append(f"Isolated tasks found: {isolated_nodes}")
        efficiency *= 0.9
    
    # Calculate overall quality
    weights = [0.3, 0.25, 0.25, 0.2]  # completeness, coherence, efficiency, feasibility
    overall_quality = sum(w * s for w, s in zip(weights, [completeness, coherence, efficiency, feasibility]))
    
    return WorkflowQualityMetrics(
        completeness_score=round(completeness, 3),
        coherence_score=round(coherence, 3),
        efficiency_score=round(efficiency, 3),
        feasibility_score=round(feasibility, 3),
        overall_quality=round(overall_quality, 3),
        validation_errors=validation_errors,
        warnings=warnings,
        suggestions=suggestions
    )


def estimate_execution_metrics(workflow: Workflow) -> ExecutionEstimate:
    subtasks_with_modes = [
        {"id": st.id, "desc": st.description, "mode": st.mode} 
        for st in workflow.subtasks
    ]
    
    # Get parallel block info
    parallel_blocks = {}
    for node in workflow.dag.nodes():
        if 'parallel_level' in workflow.dag.nodes[node]:
            level = workflow.dag.nodes[node]['parallel_level']
            if level not in parallel_blocks:
                parallel_blocks[level] = []
            parallel_blocks[level].append(node)
    
    try:
        execution_analysis = json.loads(gpt(
            EXECUTION_ESTIMATION_TEMPLATE.format(
                subtasks_with_modes=json.dumps(subtasks_with_modes, ensure_ascii=False),
                dependencies=list(workflow.dag.edges()),
                parallel_blocks=parallel_blocks
            ),
            system="Return valid JSON only.",
            temperature=0.1
        ))
        
        task_estimates = execution_analysis.get("task_estimates", {})
        critical_path_time = execution_analysis.get("critical_path_time", 1.0)
        total_sequential_time = execution_analysis.get("total_sequential_time", len(workflow.subtasks) * 0.5)
        estimated_parallel_time = execution_analysis.get("estimated_parallel_time", critical_path_time)
        bottlenecks = execution_analysis.get("bottlenecks", [])
        cost_factors = execution_analysis.get("cost_factors", {"compute": 1.0, "storage": 0.1, "network": 0.1})
        
    except Exception as e:
        # Fallback estimation
        n_tasks = len(workflow.subtasks)
        
        # Estimate based on task modes and complexity
        wide_tasks = [st for st in workflow.subtasks if st.mode == "WIDE"]
        deep_tasks = [st for st in workflow.subtasks if st.mode == "DEEP"]
        
        # WIDE tasks typically take longer due to information gathering
        wide_time = len(wide_tasks) * 1.5  # hours
        deep_time = len(deep_tasks) * 1.0   # hours
        
        total_sequential_time = wide_time + deep_time
        
        # Estimate parallel efficiency
        try:
            critical_path = nx.algorithms.dag.longest_path(workflow.dag)
            critical_path_time = len(critical_path) * (total_sequential_time / n_tasks) if n_tasks > 0 else 1.0
        except:
            critical_path_time = total_sequential_time * 0.7
        
        estimated_parallel_time = critical_path_time
        bottlenecks = ["Estimation fallback - manual review recommended"]
        cost_factors = {"compute": n_tasks * 0.5, "storage": 0.1, "network": len(wide_tasks) * 0.2}
    
    # Calculate savings from parallelization
    parallel_savings = max(0, total_sequential_time - estimated_parallel_time)
    
    # Resource requirements estimation
    resource_requirements = {
        "cpu_cores": min(len(workflow.subtasks), max(len(parallel_blocks.get(level, [])) for level in parallel_blocks)) if parallel_blocks else 1,
        "memory_gb": len(workflow.subtasks) * 2,  # 2GB per task estimate
        "storage_gb": len([st for st in workflow.subtasks if st.mode == "WIDE"]) * 10  # 10GB per WIDE task
    }
    
    return ExecutionEstimate(
        estimated_total_time=round(estimated_parallel_time, 2),
        critical_path_time=round(critical_path_time, 2),
        parallel_time_savings=round(parallel_savings, 2),
        resource_requirements=resource_requirements,
        cost_estimate=cost_factors,
        bottlenecks=bottlenecks
    )


def diagnose_workflow_issues(workflow: Workflow, quality_metrics: WorkflowQualityMetrics, 
                           complexity_metrics: ComplexityMetrics) -> Dict[str, List[str]]:
    issues = {
        "critical_issues": [],
        "performance_issues": [],
        "design_issues": [],
        "recommendations": []
    }
    
    # Critical issues
    if quality_metrics.overall_quality < 0.5:
        issues["critical_issues"].append("Overall workflow quality is poor - consider redesigning")
    
    if complexity_metrics.requires_replanning:
        issues["critical_issues"].append("Complexity analysis suggests re-planning required")
    
    if quality_metrics.validation_errors:
        issues["critical_issues"].extend(quality_metrics.validation_errors)
    
    # Performance issues
    if complexity_metrics.parallel_efficiency < 0.3:
        issues["performance_issues"].append("Poor parallelization - workflow is too sequential")
    
    if complexity_metrics.resource_conflict_factor > 0.5:
        issues["performance_issues"].append("High resource conflicts detected")
    
    if len(workflow.subtasks) > 12:
        issues["performance_issues"].append("Workflow might be too granular - consider consolidating tasks")
    
    # Design issues
    if complexity_metrics.mode_heterogeneity > 0.8:
        issues["design_issues"].append("High mode heterogeneity - consider grouping similar tasks")
    
    if quality_metrics.coherence_score < 0.6:
        issues["design_issues"].append("Workflow dependencies are not well-structured")
    
    # Recommendations
    if complexity_metrics.critical_path_factor > 1.5:
        issues["recommendations"].append("Consider breaking down critical path tasks")
    
    if quality_metrics.efficiency_score < 0.7:
        issues["recommendations"].append("Optimize task dependencies for better parallelization")
    
    if complexity_metrics.uncertainty_score > 0.7:
        issues["recommendations"].append("Add intermediate checkpoints due to high uncertainty")
    
    return issues
