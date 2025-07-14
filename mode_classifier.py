# mode_classifier.py
import json
import re
from typing import List, Dict
import networkx as nx

from models import SubTask, ModeResult
from utils import gpt


# ------------ Stage 3: Wide vs Deep classification per sub‑task ---------- #

WIDE_KW = {"research", "survey", "compare", "gather", "various", "broad", "explore", "investigate", "collect", "multiple", "diverse", "extensive", "search", "find", "discover", "review", "study", "examine", "scan", "browse", "lookup", "fetch", "retrieve", "compile", "aggregate"}
DEEP_KW = {"analyze", "solve", "create", "optimize", "reason", "critique", "develop", "calculate", "design", "implement", "construct", "derive", "synthesize", "formulate", "engineer", "build", "process", "compute", "evaluate", "assess", "determine", "conclude", "infer", "deduce"}

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


def classify_mode(st: SubTask, main_task_context: str = "") -> ModeResult:
    # Initial keyword-based analysis
    tokens = set(re.findall(r"[A-Za-z]+", st.description.lower()))
    wide_score = len(tokens & WIDE_KW)
    deep_score = len(tokens & DEEP_KW)
    
    # Contextual LLM analysis
    try:
        context_analysis = json.loads(gpt(
            CONTEXTUAL_ANALYSIS_TEMPLATE.format(
                description=st.description, 
                main_task=main_task_context
            ),
            system="Return valid JSON only.", 
            temperature=0.1
        ))
    except:
        # Fallback to simple classification
        if wide_score > deep_score:
            return ModeResult("WIDE", 0.6, False)
        elif deep_score > wide_score:
            return ModeResult("DEEP", 0.6, False)
        else:
            return ModeResult("WIDE", 0.5, False)
    
    primary_mode = context_analysis.get("primary_mode", "WIDE")
    confidence = context_analysis.get("confidence", 0.5)
    is_hybrid = context_analysis.get("is_hybrid", False)
    
    result = ModeResult(
        primary_mode=primary_mode,
        confidence=confidence,
        is_hybrid=is_hybrid,
        secondary_mode=context_analysis.get("secondary_mode"),
        information_requirements=context_analysis.get("information_requirements", []),
        processing_complexity=context_analysis.get("processing_complexity", "medium")
    )
    
    # Handle hybrid tasks
    if is_hybrid and confidence > 0.7:
        try:
            hybrid_analysis = json.loads(gpt(
                HYBRID_ANALYSIS_TEMPLATE.format(
                    description=st.description,
                    initial_analysis=json.dumps(context_analysis, ensure_ascii=False)
                ),
                system="Return valid JSON only.",
                temperature=0.1
            ))
            
            result.execution_strategy = hybrid_analysis.get("execution_strategy")
            result.phase_breakdown = hybrid_analysis.get("phase_breakdown", [])
            
        except:
            # Fallback: default to sequential strategy
            result.execution_strategy = "sequential_wide_then_deep"
    
    return result


def determine_optimal_mode_sequence(subtasks: List[SubTask], dag: nx.DiGraph, main_task: str) -> Dict:
    mode_results = {}
    execution_plan = {
        "sequential_phases": [],
        "parallel_blocks": {},
        "hybrid_tasks": [],
        "mode_transitions": []
    }
    
    # Classify all subtasks
    for st in subtasks:
        mode_result = classify_mode(st, main_task)
        mode_results[st.id] = mode_result
        st.mode = mode_result.primary_mode  # Set basic mode for compatibility
        
        if mode_result.is_hybrid:
            execution_plan["hybrid_tasks"].append({
                "task_id": st.id,
                "strategy": mode_result.execution_strategy,
                "phases": mode_result.phase_breakdown
            })
    
    # Analyze parallel blocks for mode optimization
    for node in dag.nodes():
        if 'parallel_level' in dag.nodes[node]:
            level = dag.nodes[node]['parallel_level']
            if level not in execution_plan["parallel_blocks"]:
                execution_plan["parallel_blocks"][level] = {
                    "tasks": [],
                    "dominant_mode": None,
                    "mixed_mode": False
                }
            
            task_mode = mode_results[node].primary_mode
            execution_plan["parallel_blocks"][level]["tasks"].append({
                "task_id": node,
                "mode": task_mode,
                "confidence": mode_results[node].confidence
            })
    
    # Determine dominant mode for each parallel block
    for level, block_info in execution_plan["parallel_blocks"].items():
        modes = [task["mode"] for task in block_info["tasks"]]
        wide_count = modes.count("WIDE")
        deep_count = modes.count("DEEP")
        
        if wide_count > deep_count:
            block_info["dominant_mode"] = "WIDE"
        elif deep_count > wide_count:
            block_info["dominant_mode"] = "DEEP"
        else:
            block_info["dominant_mode"] = "MIXED"
            block_info["mixed_mode"] = True
    
    return execution_plan
