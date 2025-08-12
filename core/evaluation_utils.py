# core/evaluation_utils.py
import statistics
import json
from typing import Dict, Any, Optional, List, Set, Tuple
import networkx as nx
import datetime

from models import SubTask


class MetricsCalculator:
    """Unified metrics calculation for workflows and DAGs."""
    
    @staticmethod
    def structural_metrics(pred: nx.DiGraph, gold: Optional[nx.DiGraph] = None) -> Dict[str, Any]:
        """Calculate structural metrics: Num Nodes/Edges, Density, Average Degree, Parallel Efficiency."""
        # Filter out group nodes for fair comparison
        actual_nodes = [n for n in pred.nodes() if not pred.nodes[n].get('is_group', False)]
        actual_pred = pred.subgraph(actual_nodes).copy()
        
        # Core structural metrics only
        metrics: Dict[str, Any] = {
            "num_nodes": len(actual_nodes),
            "num_edges": actual_pred.number_of_edges(),
            "density": nx.density(actual_pred) if len(actual_nodes) > 1 else 0.0,
            "average_degree": (
                sum(dict(actual_pred.degree()).values()) / len(actual_nodes)
                if len(actual_nodes) > 0 else 0.0
            ),
        }

        # Parallel Efficiency: 1 - critical_path_length/total_nodes
        if nx.is_directed_acyclic_graph(actual_pred) and len(actual_nodes) > 0:
            try:
                critical_path_length = len(nx.dag_longest_path(actual_pred))
                metrics["parallel_efficiency"] = 1.0 - (critical_path_length / len(actual_nodes))
            except (nx.NetworkXNoPath, ZeroDivisionError):
                metrics["parallel_efficiency"] = 0.0
        else:
            metrics["parallel_efficiency"] = 0.0

        return metrics
    
    @staticmethod
    def execution_metrics(dag: nx.DiGraph, durations: Dict[str, float], 
                         resources: Optional[Dict[str, Any]] = None, *, 
                         exclude_group_nodes: bool = True) -> Dict[str, Any]:
        """Calculate execution-oriented quality measures (kept for compatibility)."""
        # This method is kept for compatibility but returns minimal metrics
        # since execution details are now handled in structural_metrics
        return {
            "parallel_efficiency": 0.0  # Placeholder for compatibility
        }
    
    @staticmethod
    def complexity_metrics(workflow_data: Dict[str, Any], llm_client=None, main_task: str = "") -> Dict[str, float]:
        """Calculate complexity metrics using LLM analysis."""
        subtasks = workflow_data.get('subtasks', [])
        dag = workflow_data.get('dag')
        
        if not subtasks:
            return {
                "domain_complexity": 5.0,
                "coordination_complexity": 5.0,
                "computational_complexity": 5.0
            }
        
        # If no LLM client provided, use heuristic fallback
        if llm_client is None:
            return {
                "domain_complexity": min(10.0, len(subtasks) * 0.5 + 3.0),
                "coordination_complexity": min(10.0, dag.number_of_edges() * 0.1 + 2.0) if dag else 3.0,
                "computational_complexity": min(10.0, len(subtasks) * 0.8 + 2.0)
            }
        
        # Use LLM for complexity analysis
        try:
            from prompts import PromptManager
            subtasks_info = [{"id": st.id if hasattr(st, 'id') else f"S{i}", 
                            "desc": st.description if hasattr(st, 'description') else str(st)} 
                           for i, st in enumerate(subtasks)]
            
            prompt = PromptManager.format_complexity_analysis_prompt(
                main_task,
                json.dumps(subtasks_info, ensure_ascii=False)
            )
            
            analysis = llm_client.call_json(prompt)
            
            return {
                "domain_complexity": analysis.get("domain_complexity", 5.0),
                "coordination_complexity": analysis.get("coordination_complexity", 5.0),
                "computational_complexity": analysis.get("computational_complexity", 5.0)
            }
            
        except Exception:
            # Fallback to heuristic if LLM fails
            return {
                "domain_complexity": min(10.0, len(subtasks) * 0.5 + 3.0),
                "coordination_complexity": min(10.0, dag.number_of_edges() * 0.1 + 2.0) if dag else 3.0,
                "computational_complexity": min(10.0, len(subtasks) * 0.8 + 2.0)
            }
    
    @staticmethod
    def _edge_set(g: nx.DiGraph) -> Set[Tuple[str, str]]:
        """Return set of directed edges as (u, v) tuples."""
        return {(u, v) for u, v in g.edges()}
    
    # Removed _critical_path_time method as it's no longer needed


class QualityAssessor:
    """LLM-based quality assessment for workflows."""
    
    @staticmethod
    def assess_workflow_quality(workflow_data: Dict[str, Any], 
                               subtasks: List[SubTask],
                               dag: nx.DiGraph,
                               llm_client=None,
                               main_task: str = "") -> Dict[str, float]:
        """Assess workflow quality using LLM analysis."""
        
        # If no LLM client provided, use heuristic fallback
        if llm_client is None:
            print("         ⚠️  Using null fallback (no LLM client)")
            return QualityAssessor._heuristic_quality_assessment(subtasks, dag)
        
        # Use LLM for quality assessment
        try:
            from prompts import PromptManager
            
            # Prepare subtasks and dependencies for LLM
            subtasks_info = [{"id": st.id if hasattr(st, 'id') else f"S{i}", 
                            "desc": st.description if hasattr(st, 'description') else str(st)} 
                           for i, st in enumerate(subtasks)]
            dependencies = list(dag.edges()) if dag else []
            
            # Format quality assessment prompt
            prompt = PromptManager.format_quality_assessment_prompt(
                main_task,
                json.dumps(subtasks_info, ensure_ascii=False),
                str(dependencies)
            )
            
            analysis = llm_client.call_json(prompt)
            
            # Extract scores with fallback to defaults
            completeness = analysis.get("completeness_score", 0.5)
            coherence = analysis.get("coherence_score", 0.5)
            efficiency = analysis.get("efficiency_score", 0.5)
            feasibility = analysis.get("feasibility_score", 0.5)
            
            print(f"         ✅ LLM quality scores: C:{completeness:.3f} Coh:{coherence:.3f} E:{efficiency:.3f} F:{feasibility:.3f}")
            
            return {
                "completeness_score": round(completeness, 3),
                "coherence_score": round(coherence, 3),
                "efficiency_score": round(efficiency, 3),
                "feasibility_score": round(feasibility, 3)
            }
            
        except Exception as e:
            # Fallback to heuristic if LLM fails
            print(f"         ⚠️  LLM quality assessment failed, using null fallback: {e}")
            return QualityAssessor._heuristic_quality_assessment(subtasks, dag)
    
    @staticmethod
    def _heuristic_quality_assessment(subtasks: List[SubTask], dag: nx.DiGraph) -> Dict[str, float]:
        """Fallback heuristic quality assessment."""
        # When LLM is unavailable, return null values to indicate missing data
        # All quality metrics should ideally come from LLM analysis
        completeness = None 
        coherence = None     
        efficiency = None    # Different from structural parallel efficiency
        feasibility = None
        
        return {
            "completeness_score": completeness,
            "coherence_score": coherence,
            "efficiency_score": efficiency,
            "feasibility_score": feasibility
        }
    
    @staticmethod
    def generate_recommendations(per_approach_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommendations based on simplified evaluation metrics."""
        
        if not per_approach_metrics:
            return {"error": "No approaches to evaluate"}
        
        # Simplified scoring weights focusing on our core metrics
        use_case_weights = {
            "structural_focused": {
                "num_nodes": 0.1,
                "num_edges": 0.1,
                "density": 0.2,
                "average_degree": 0.2,
                "parallel_efficiency": 0.4
            },
            "quality_focused": {
                "completeness_score": 0.3,
                "coherence_score": 0.25,
                "efficiency_score": 0.25,
                "feasibility_score": 0.2
            },
            "complexity_focused": {
                "domain_complexity": -0.33,  # Lower is better
                "coordination_complexity": -0.33,
                "computational_complexity": -0.34
            }
        }
        
        recommendations = {}
        
        for use_case, weights in use_case_weights.items():
            best_approach = None
            best_score = float('-inf')
            
            scores = {}
            for approach, metrics in per_approach_metrics.items():
                score = 0
                for metric, weight in weights.items():
                    value = metrics.get(metric, 0)
                    if value is not None:
                        score += value * weight
                scores[approach] = score
                
                if score > best_score:
                    best_score = score
                    best_approach = approach
            
            recommendations[use_case] = {
                "recommended_approach": best_approach,
                "score": best_score,
                "all_scores": scores,
                "reasoning": QualityAssessor._generate_reasoning(best_approach, use_case, 
                                                               per_approach_metrics[best_approach] if best_approach else {})
            }
        
        return recommendations
    
    @staticmethod
    def _generate_reasoning(approach: str, use_case: str, metrics: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for recommendation."""
        
        if not approach:
            return "No suitable approach found"
        
        reasoning_parts = [f"{approach} is recommended for {use_case} use case because:"]
        
        if use_case == "structural_focused":
            parallel_eff = metrics.get("parallel_efficiency", 0)
            density = metrics.get("density", 0)
            reasoning_parts.append(f"- High parallel efficiency ({parallel_eff:.3f})")
            reasoning_parts.append(f"- Good graph density ({density:.3f})")
            
        elif use_case == "quality_focused":
            completeness = metrics.get("completeness_score", 0)
            coherence = metrics.get("coherence_score", 0)
            efficiency = metrics.get("efficiency_score", 0)
            feasibility = metrics.get("feasibility_score", 0)
            
            reasoning_parts.append(f"- High completeness ({completeness:.3f})")
            reasoning_parts.append(f"- Good coherence ({coherence:.3f})")
            reasoning_parts.append(f"- Efficient structure ({efficiency:.3f})")
            reasoning_parts.append(f"- High feasibility ({feasibility:.3f})")
            
        elif use_case == "complexity_focused":
            domain = metrics.get("domain_complexity", 0)
            coordination = metrics.get("coordination_complexity", 0)
            computational = metrics.get("computational_complexity", 0)
            
            reasoning_parts.append(f"- Low domain complexity ({domain:.1f})")
            reasoning_parts.append(f"- Low coordination complexity ({coordination:.1f})")
            reasoning_parts.append(f"- Low computational complexity ({computational:.1f})")
        
        return "\n".join(reasoning_parts)


class ComparisonEvaluator:
    """Unified comparison evaluation for different DAG approaches."""
    
    def __init__(self, gold_dags: Optional[Dict[str, nx.DiGraph]] = None,
                 durations_db: Optional[Dict[str, Dict[str, float]]] = None,
                 resources_db: Optional[Dict[str, Dict[str, Any]]] = None):
        self.gold_dags = gold_dags or {}
        self.durations_db = durations_db or {}
        self.resources_db = resources_db or {}
    
    def evaluate_comparison_results(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate results from compare_dag_approaches() with simplified metrics."""
        
        # Extract task_id from successful approach
        task_id = None
        for approach_name, result in comparison_results.items():
            if approach_name != "comparison_summary" and result.get("success", False):
                task_id = result.get("metrics", {}).get("task_id")
                break
        
        # Evaluate each approach with simplified metrics
        per_approach = {}
        for approach_name, result in comparison_results.items():
            if approach_name == "comparison_summary" or not result.get("success", False):
                continue
            
            dag: nx.DiGraph = result["dag"]
            
            # Get simplified structural metrics only
            structural = MetricsCalculator.structural_metrics(dag)
            
            # Get workflow evaluation if available
            workflow_eval = result.get("workflow_evaluation", {})
            quality_metrics = workflow_eval.get("quality_metrics", {})
            complexity_metrics = workflow_eval.get("complexity_metrics", {})
            
            # Combine all metrics
            per_approach[approach_name] = {
                **structural,
                **quality_metrics,
                **complexity_metrics
            }
        
        # Generate aggregate statistics for numeric metrics only
        aggregate_summary = {}
        if per_approach:
            keys = next(iter(per_approach.values())).keys()
            for k in keys:
                vals = [v[k] for v in per_approach.values() if isinstance(v[k], (int, float))]
                if not vals:
                    continue
                aggregate_summary[k] = {
                    "mean": statistics.mean(vals),
                    "best": max(vals) if k != "domain_complexity" and k != "coordination_complexity" and k != "computational_complexity" else min(vals),
                }
        
        # Generate recommendations
        recommendations = QualityAssessor.generate_recommendations(per_approach)
        
        return {
            "per_approach": per_approach,
            "aggregate_summary": aggregate_summary,
            "recommendations": recommendations,
            "metadata": {
                "task_id": task_id,
                "evaluation_timestamp": datetime.datetime.now().isoformat(),
                "approaches_evaluated": list(per_approach.keys())
            }
        }
