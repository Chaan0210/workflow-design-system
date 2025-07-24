# prompts.py

# ------------------ From complexity_analyzer.py ------------------ #

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

# ------------------ From dag_builder.py ------------------ #

EDGE_TEMPLATE = """
For the following two sub‑tasks from the same parent task, answer with STRICT JSON "yes" or "no" only:

Sub‑task A: "{a}"
Sub‑task B: "{b}"

Question: Can B start before A is completed?
"""

RESOURCE_DEPENDENCY_TEMPLATE = """
Analyze if these two sub-tasks have any shared resource dependencies or conflicts.
Return JSON with "resource_conflict": true/false and "shared_resources": [list of shared resources].

Sub‑task A: "{a}"
Sub‑task B: "{b}"
"""

CONFIDENCE_TEMPLATE = """
Rate your confidence (0.0 to 1.0) in the dependency relationship between these tasks:

Sub‑task A: "{a}"  
Sub‑task B: "{b}"

Return only a number between 0.0 and 1.0.
"""

# ------------------ From decomposition.py ------------------ #

DECOMPOSE_TEMPLATE = """
You are an intelligent workflow planner. Your first step is to decompose the given task into a set of necessary, self-contained sub-tasks that a team of autonomous agents could handle.
Focus only on identifying and listing the sub-tasks. The dependency analysis and agent assignment will be handled in a later step.

Task Requirements:
{task}

Return a JSON list of objects, where each object has an "id" and a "desc" field. For example:
[
  {{ "id": "S1", "desc": "Description of the first sub-task." }},
  {{ "id": "S2", "desc": "Description of the second sub-task." }}
]
"""

# ------------------ From mode_classifier.py ------------------ #

CONTEXTUAL_ANALYSIS_TEMPLATE = """
Analyze the following sub-task and determine its primary nature:

Task: "{description}"
Context: This is part of a larger workflow for: "{main_task}"

Consider:
1. Does this task primarily require gathering diverse external information? (WIDE)
2. Does this task primarily require deep reasoning/analysis of limited information? (DEEP)  
3. What is the information processing pattern?

Return JSON with:
{{
    "primary_mode": "WIDE" or "DEEP",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "is_hybrid": true/false,
    "secondary_mode": "WIDE"/"DEEP"/null,
    "information_requirements": ["requirement1", "requirement2", ...],
    "processing_complexity": "low"/"medium"/"high"
}}
"""

HYBRID_ANALYSIS_TEMPLATE = """
This task appears to have both WIDE and DEEP characteristics. Determine the optimal execution strategy:

Task: "{description}"
Initial analysis: {initial_analysis}

Return JSON with:
{{
    "execution_strategy": "sequential_wide_then_deep" / "sequential_deep_then_wide" / "parallel_hybrid" / "dynamic_switching",
    "phase_breakdown": [
        {{"phase": 1, "mode": "WIDE/DEEP", "description": "what to do in this phase"}},
        ...
    ],
    "transition_criteria": "when/how to switch between modes"
}}
"""

# ------------------ From validator.py ------------------ #

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