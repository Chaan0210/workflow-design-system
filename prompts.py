# prompts.py

from __future__ import annotations

class WorkflowPrompts:
    """Core workflow planning prompts (decomposition, mode tagging, etc.)."""

    DECOMPOSITION = (
        """\
### SYSTEM
You are a world expert at making efficient plans.
Think carefully, de‑duplicate similar steps, and ensure coverage of the main
objective.
Respond with **ONLY** valid JSON — no surrounding prose.

```json schema
[
  {{"id": "S1", "desc": "<string>"}},
  {{"id": "S2", "desc": "<string>"}},
  ...
]
```

### USER
Task Requirements:
{task}
"""
    )

    MODE_CLASSIFICATION = (
        """\
### SYSTEM
Classify the cognitive *mode* of the following sub‑task.
Internal reasoning is allowed, but **only** the final JSON must be output.
Follow the schema strictly.

```json schema
{{
  "primary_mode": "WIDE" | "DEEP",
  "confidence": 0.0‑1.0,
  "reasoning": "<max 200 chars>",
  "is_hybrid": true | false,
  "secondary_mode": "WIDE" | "DEEP" | null,
  "information_requirements": ["<string>", …],
  "processing_complexity": "low" | "medium" | "high"
}}
```

### USER
Main Task: "{main_task}"
Sub‑Task: "{description}"
"""
    )

    COMPLEXITY_ANALYSIS = (
        """\
### SYSTEM
Estimate complexity / uncertainty factors for the task bundle. Think rigorously,
but output **only** JSON adhering to the schema.

```json schema
{{
  "domain_complexity": 1‑10,
  "coordination_complexity": 1‑10,
  "computational_complexity": 1‑10,
  "temporal_uncertainty": 1‑10,
  "resource_uncertainty": 1‑10,
  "outcome_uncertainty": 1‑10,
  "overall_uncertainty": 0.0‑1.0,
  "requires_replanning": true | false,
  "risk_factors": ["<string>", …],
  "mitigation_strategies": ["<string>", …]
}}
```

### USER
Main Task: "{task}"
Sub‑Tasks JSON: {subtasks}
"""
    )

    QUALITY_VALIDATION = (
        """\
### SYSTEM
Critique the workflow design on four axes. Produce scores and diagnostics in the
JSON schema below.

```json schema
{{
  "completeness_score": 0.0‑1.0,
  "coherence_score": 0.0‑1.0,
  "efficiency_score": 0.0‑1.0,
  "feasibility_score": 0.0‑1.0,
  "validation_errors": ["<string>", …],
  "warnings": ["<string>", …],
  "suggestions": ["<string>", …],
  "missing_subtasks": ["<string>", …],
  "redundant_subtasks": ["<string>", …],
  "problematic_dependencies": ["<string>", …]
}}
```

### USER
Main Task: "{main_task}"
Sub‑Tasks JSON: {subtasks}
Dependencies: {dependencies}
"""
    )


class DependencyPrompts:
    """Prompts for pair‑wise dependency and resource analysis."""

    DEPENDENCY_ANALYSIS = (
        """\
### SYSTEM
Determine if B depends on A. Return JSON only.

```json schema
{{
  "dependent": true | false,
  "confidence": 0.0‑1.0,
  "reason": "<max 120 chars>"
}}
```

### USER
Original Task Context: "{original_task}"
A: "{a}"
B: "{b}"
"""
    )

    RESOURCE_CONFLICT = (
        """\
### SYSTEM
Detect concrete *resource* conflicts between tasks A & B.

```json schema
{{
  "resource_conflict": true | false,
  "shared_resources": ["<string>", …]
}}
```

### USER
Original Task Context: "{original_task}"
A: "{a}"
B: "{b}"
"""
    )

    CROSS_TREE_DEPENDENCY = (
        """\
### SYSTEM
Two subtasks belong to different branches. Decide if A must finish before B.
Return JSON only.

```json schema
{{"dependent": true | false, "confidence": 0.0‑1.0, "reasoning": "<120 chars>"}}
```

### USER
Task A: {task_a_desc}
Task B: {task_b_desc}
"""
    )

    RESOURCE_CONFLICT_ANALYSIS = (
        """\
### SYSTEM
Identify overlapping resources.

```json schema
{{
  "has_conflict": true | false,
  "shared_resources": ["<string>", …],
  "conflict_severity": "low" | "medium" | "high"
}}
```

### USER
Task A: {task_a_desc}
Task B: {task_b_desc}
"""
    )


class ApproachSpecificPrompts:
    """Prompts tailored to particular DAG‑building strategies."""

    HIERARCHICAL_ORGANIZATION = (
        """\
### SYSTEM
Group the existing subtasks into a logical tree. **Do not** invent new IDs.
Output JSON that matches the schema; no extra commentary.

```json schema
{{
  "id": "ROOT",
  "desc": "<string>",
  "children": [
    {{
      "id": "<group_name_or_subtask_id>",
      "desc": "<string>", 
      "children": [
        {{"id": "S1", "desc": "<string>", "children": []}},
        {{"id": "S2", "desc": "<string>", "children": []}}
      ]
    }}
  ]
}}
```

### USER
Original Task: {task}
Existing Subtasks:
{subtask_list}
"""
    )

    HIERARCHICAL_DECOMPOSE = (
        """\
### SYSTEM
You are an intelligent workflow planner. Decompose the given task into a hierarchical tree structure where:

- Each task can have sub-tasks
- Sub-tasks at the same level can potentially run in parallel
- Parent tasks depend on their children completing first
- The tree structure represents natural dependency relationships

Return a JSON tree structure where each node has:
- "id": unique identifier
- "desc": task description
- "children": list of child tasks (empty list if leaf node)

```json schema
{{
  "id": "ROOT",
  "desc": "Main task description",
  "children": [
    {{
      "id": "S1",
      "desc": "First major subtask",
      "children": [
        {{"id": "S1.1", "desc": "First sub-subtask", "children": []}},
        {{"id": "S1.2", "desc": "Second sub-subtask", "children": []}}
      ]
    }},
    {{
      "id": "S2",
      "desc": "Second major subtask",
      "children": []
    }}
  ]
}}
```

### USER
Task Requirements:
{task}
"""
    )

    MATRIX_GENERATION = (
        """\
### SYSTEM
Create a full adjacency matrix of strict dependencies. Think about data flow,
resource usage, and logical order, but output only JSON.

```json schema
{{
  "matrix": [[0|1, …]],
  "confidence_matrix": [[0.0‑1.0, …]],
  "task_order": ["<SubtaskID>", …]
}}
```

### USER
Original Task: {original_task}
Subtasks:
{subtask_list}
"""
    )

    ENHANCED_MATRIX_GENERATION = (
        """\
### SYSTEM
Generate a dependency matrix that *maximises parallel execution* while
maintaining correctness. Follow the five optimisation rules below FIRST, then
return JSON only.

```json schema
{{
  "matrix": [[0|1, …]],
  "confidence_matrix": [[0.0‑1.0, …]],
  "task_order": ["<SubtaskID>", …],
  "parallel_blocks": [["<SubtaskID>", …], …]
}}
```

### OPTIMISATION RULES
1. Independent information‑gathering tasks → **no** dependencies.
2. Only impose an edge if *output‑to‑input* is mandatory.
3. Calculations that share inputs run in parallel.
4. Final aggregation / formatting depends on all calculation tasks.
5. Prefer fan‑in over deep chains.

### USER
Original Task: {original_task}
Subtasks:
{subtask_list}
"""
    )


class PromptManager:
    """Centralized prompt management with formatting methods."""
    
    @staticmethod
    def format_decomposition_prompt(task: str) -> str:
        """Format task decomposition prompt."""
        return WorkflowPrompts.DECOMPOSITION.format(task=task)
    
    @staticmethod
    def format_mode_classification_prompt(description: str, main_task: str) -> str:
        """Format mode classification prompt."""
        return WorkflowPrompts.MODE_CLASSIFICATION.format(
            description=description, main_task=main_task
        )
    
    @staticmethod
    def format_complexity_analysis_prompt(task: str, subtasks: str) -> str:
        """Format complexity analysis prompt."""
        return WorkflowPrompts.COMPLEXITY_ANALYSIS.format(task=task, subtasks=subtasks)
    
    @staticmethod
    def format_quality_validation_prompt(main_task: str, subtasks: str, dependencies: str) -> str:
        """Format quality validation prompt."""
        return WorkflowPrompts.QUALITY_VALIDATION.format(
            main_task=main_task, subtasks=subtasks, dependencies=dependencies
        )
    
    @staticmethod
    def format_quality_assessment_prompt(main_task: str, subtasks: str, dependencies: str) -> str:
        """Format quality assessment prompt."""
        return WorkflowPrompts.QUALITY_VALIDATION.format(
            main_task=main_task, subtasks=subtasks, dependencies=dependencies
        )
    
    @staticmethod
    def format_dependency_prompt(original_task: str, task_a: str, task_b: str) -> str:
        """Format dependency analysis prompt."""
        return DependencyPrompts.DEPENDENCY_ANALYSIS.format(
            original_task=original_task, a=task_a, b=task_b
        )
    
    @staticmethod
    def format_resource_conflict_prompt(original_task: str, task_a: str, task_b: str) -> str:
        """Format resource conflict analysis prompt."""
        return DependencyPrompts.RESOURCE_CONFLICT.format(
            original_task=original_task, a=task_a, b=task_b
        )
    
    @staticmethod
    def format_cross_tree_dependency_prompt(task_a_desc: str, task_b_desc: str) -> str:
        """Format cross-tree dependency prompt."""
        return DependencyPrompts.CROSS_TREE_DEPENDENCY.format(
            task_a_desc=task_a_desc, task_b_desc=task_b_desc
        )
    
    @staticmethod
    def format_resource_conflict_analysis_prompt(task_a_desc: str, task_b_desc: str) -> str:
        """Format resource conflict analysis prompt."""
        return DependencyPrompts.RESOURCE_CONFLICT_ANALYSIS.format(
            task_a_desc=task_a_desc, task_b_desc=task_b_desc
        )
    
    @staticmethod
    def format_hierarchical_organization_prompt(task: str, subtask_list: str) -> str:
        """Format hierarchical organization prompt."""
        return ApproachSpecificPrompts.HIERARCHICAL_ORGANIZATION.format(
            task=task, subtask_list=subtask_list
        )
    
    @staticmethod
    def format_hierarchical_decompose_prompt(task: str) -> str:
        """Format hierarchical decomposition prompt."""
        return ApproachSpecificPrompts.HIERARCHICAL_DECOMPOSE.format(task=task)
    
    @staticmethod
    def format_matrix_generation_prompt(original_task: str, subtask_list: str) -> str:
        """Format matrix generation prompt."""
        return ApproachSpecificPrompts.MATRIX_GENERATION.format(
            original_task=original_task, subtask_list=subtask_list
        )
    
    @staticmethod
    def format_enhanced_matrix_generation_prompt(original_task: str, subtask_list: str) -> str:
        """Format enhanced matrix generation prompt with parallel optimization."""
        return ApproachSpecificPrompts.ENHANCED_MATRIX_GENERATION.format(
            original_task=original_task, subtask_list=subtask_list
        )


# Legacy compatibility - maintain the existing templates for backward compatibility
DEPENDENCY_TEMPLATE = DependencyPrompts.DEPENDENCY_ANALYSIS
RESOURCE_CONFLICT_TEMPLATE = DependencyPrompts.RESOURCE_CONFLICT
DECOMPOSITION_TEMPLATE = WorkflowPrompts.DECOMPOSITION
MODE_CLASSIFICATION_TEMPLATE = WorkflowPrompts.MODE_CLASSIFICATION
COMPLEXITY_ANALYSIS_TEMPLATE = WorkflowPrompts.COMPLEXITY_ANALYSIS
QUALITY_VALIDATION_TEMPLATE = WorkflowPrompts.QUALITY_VALIDATION
HIERARCHICAL_DECOMPOSE_TEMPLATE = ApproachSpecificPrompts.HIERARCHICAL_DECOMPOSE
