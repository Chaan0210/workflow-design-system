# prompts.py — v2 (2025‑08‑06)
#
# Centralized prompt templates tuned for *structured‑output reliability* and
# *planning‑quality*, informed by recent research:
#   • Schema‑based prompting (SchemaBench, 2025)
#   • Few‑shot / instruction synergy (Schick et al., 2022; Shelf.io blog, 2024)
#   • Constrained JSON generation guides (OpenAI Structured Outputs, 2025)
#
# Key improvements vs v1:
# 1. Single‑source of truth for JSON schema → every prompt embeds the exact
#    JSON skeleton the model must return; no prose allowed.
# 2. Strict delimiters using ```json / ``` to reduce hallucinated text.
# 3. Step‑back thinking ("Analyse step‑by‑step **internally**, do **not**
#    expose chain‑of‑thought") to encourage reasoning without leaking.
# 4. Few‑shot anchors (1 example per complex prompt) to ground the format.
# 5. Parallel‑aware guidelines in matrix generation prompt emphasise
#    independence detection heuristics.

from __future__ import annotations

class WorkflowPrompts:
    """Core workflow planning prompts (decomposition, mode tagging, etc.)."""

    DECOMPOSITION = (
        """\
### SYSTEM
You are *PlanGPT*, an expert workflow architect.
Think carefully, de‑duplicate similar steps, and ensure coverage of the main
objective. **DO NOT** perform dependency analysis here.
Respond with **ONLY** valid JSON — no surrounding prose.

```json schema
[
  {{"id": "S1", "desc": "<string>"}},
  {{"id": "S2", "desc": "<string>"}}
]
```

### USER
Task Requirements:
{task}

### EXAMPLE (few‑shot)
Input: "Plan a weekend city break in Paris for two"
Output:
```json
[
  {{"id": "S1", "desc": "Book return train/flight tickets"}},
  {{"id": "S2", "desc": "Reserve central accommodation"}},
  {{"id": "S3", "desc": "Draft 2‑day sightseeing itinerary"}},
  {{"id": "S4", "desc": "Purchase museum passes"}}
]
```

### NOW COMPLETE THE TASK ABOVE
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

### EXAMPLE
Input subtasks: S1: Find pace, S2: Find distance, S3: Calculate time
Output:
```json
{{
  "id": "ROOT",
  "desc": "Calculate running time",
  "children": [
    {{
      "id": "data_gathering",
      "desc": "Gather required data",
      "children": [
        {{"id": "S1", "desc": "Find pace", "children": []}},
        {{"id": "S2", "desc": "Find distance", "children": []}}
      ]
    }},
    {{
      "id": "S3", 
      "desc": "Calculate time",
      "children": []
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
