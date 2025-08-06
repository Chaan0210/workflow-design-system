# prompts.py - Centralized Prompt Management
"""
Centralized repository for all LLM prompts used in the dynamic workflow design system.
This module consolidates all prompt templates from across the codebase to ensure consistency,
maintainability, and easy modification of prompts.
"""

class WorkflowPrompts:
    """Core workflow planning prompts."""
    
    DECOMPOSITION = """
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

    MODE_CLASSIFICATION = """
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

    COMPLEXITY_ANALYSIS = """
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
    "mitigation_strategies": ["strategy1", "strategy2", ...]
}}
"""

    QUALITY_VALIDATION = """
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


class DependencyPrompts:
    """Prompts for dependency and resource analysis."""
    
    DEPENDENCY_ANALYSIS = """
You MUST answer in **valid JSON** only, matching this schema exactly:

{{
  "dependent": true/false,     // true if B CANNOT start before A finishes
  "confidence": 0.0-1.0,       // numeric
  "reason": "short reason"     // <= 200 chars
}}

Definition: B depends on A if B cannot start before A finishes.

Original Task Context: "{original_task}"

A: "{a}"
B: "{b}"
"""

    RESOURCE_CONFLICT = """
You MUST answer in **valid JSON** only, matching this schema exactly:

{{
  "resource_conflict": true/false,
  "shared_resources": ["..."]
}}

Original Task Context: "{original_task}"

A: "{a}"
B: "{b}"
"""

    CROSS_TREE_DEPENDENCY = """
Analyze the dependency relationship between these two subtasks from different workflow groups:

Task A: {task_a_desc}
Task B: {task_b_desc}

Does Task A need to complete before Task B can start? Consider:
- Information flow and data dependencies
- Logical sequence requirements
- Resource access needs

Return JSON:
{{
    "dependent": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""

    RESOURCE_CONFLICT_ANALYSIS = """
Analyze potential resource conflicts between these two subtasks:

Task A: {task_a_desc}
Task B: {task_b_desc}

Do these tasks compete for the same resources? Consider:
- Data sources (websites, databases, APIs)
- Computational resources
- External services
- Human attention/expertise

Return JSON:
{{
    "has_conflict": true/false,
    "shared_resources": ["list of shared resource names"],
    "conflict_severity": "low/medium/high"
}}
"""


class ApproachSpecificPrompts:
    """Approach-specific prompts for different DAG building methods."""
    
    HIERARCHICAL_ORGANIZATION = """
You are an intelligent workflow planner. Organize the given subtasks into a hierarchical tree structure.

Original Task: {task}

Existing Subtasks:
{subtask_list}

Organize these existing subtasks into a hierarchical tree where:
- ROOT is the main task
- Group related subtasks under logical parent categories
- Use ONLY the existing subtask IDs (S1, S2, S3, etc.)
- Each subtask must appear exactly once in the tree

Return ONLY a JSON tree structure where each node has:
- "id": task identifier (use ROOT for top level, then existing subtask IDs)
- "desc": task description
- "children": list of child task IDs (empty array [] for leaf nodes)

Example format:
{{
  "id": "ROOT",
  "desc": "Calculate marathon distance to moon",
  "children": [
    {{
      "id": "research_group",
      "desc": "Data gathering phase",
      "children": [
        {{"id": "S1", "desc": "Get marathon pace", "children": []}},
        {{"id": "S2", "desc": "Get moon distance", "children": []}}
      ]
    }},
    {{
      "id": "calculation_group", 
      "desc": "Calculation phase",
      "children": [
        {{"id": "S3", "desc": "Convert units", "children": []}},
        {{"id": "S4", "desc": "Calculate time", "children": []}},
        {{"id": "S5", "desc": "Convert to hours", "children": []}},
        {{"id": "S6", "desc": "Round result", "children": []}}
      ]
    }}
  ]
}}

Return ONLY the JSON structure, no other text.
"""

    MATRIX_GENERATION = """
Create a dependency adjacency matrix for these subtasks.

Original Task: {original_task}

Subtasks:
{subtask_list}

Return ONLY valid JSON in this exact format:
{{
  "matrix": [
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
  ],
  "confidence_matrix": [
    [0.0, 0.8, 0.0],
    [0.0, 0.0, 0.9],
    [0.0, 0.0, 0.0]
  ],
  "task_order": ["S1", "S2", "S3"]
}}

Rules:
- Matrix[i][j] = 1 means task i must complete before task j starts
- Matrix diagonal must be all 0s
- confidence_matrix has same dimensions with values 0.0-1.0
- task_order lists all subtask IDs in order
- Return ONLY the JSON, no comments or other text
"""


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


# Legacy compatibility - maintain the existing templates for backward compatibility
DEPENDENCY_TEMPLATE = DependencyPrompts.DEPENDENCY_ANALYSIS
RESOURCE_CONFLICT_TEMPLATE = DependencyPrompts.RESOURCE_CONFLICT
DECOMPOSITION_TEMPLATE = WorkflowPrompts.DECOMPOSITION
MODE_CLASSIFICATION_TEMPLATE = WorkflowPrompts.MODE_CLASSIFICATION
COMPLEXITY_ANALYSIS_TEMPLATE = WorkflowPrompts.COMPLEXITY_ANALYSIS
QUALITY_VALIDATION_TEMPLATE = WorkflowPrompts.QUALITY_VALIDATION