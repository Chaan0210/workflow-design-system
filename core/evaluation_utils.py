# core/evaluation_utils.py
import statistics
from typing import Dict, Any, Optional, List, Set, Tuple
import networkx as nx
import datetime

from models import SubTask


class MetricsCalculator:
    """Unified metrics calculation for workflows and DAGs."""
    
    @staticmethod
    def structural_metrics(pred: nx.DiGraph, gold: Optional[nx.DiGraph] = None) -> Dict[str, Any]:
        """Calculate graph-structural indicators."""
        # Filter out group nodes for fair comparison
        actual_nodes = [n for n in pred.nodes() if not pred.nodes[n].get('is_group', False)]
        actual_pred = pred.subgraph(actual_nodes).copy()
        
        metrics: Dict[str, Any] = {
            "num_nodes": len(actual_nodes),
            "num_edges": actual_pred.number_of_edges(),
            "is_dag": nx.is_directed_acyclic_graph(actual_pred),
            "density": nx.density(actual_pred) if len(actual_nodes) > 1 else 0.0,
            "average_degree": (
                sum(dict(actual_pred.degree()).values()) / len(actual_nodes)
                if len(actual_nodes) > 0 else 0.0
            ),
        }

        # Longest path length (counted in nodes) - use filtered graph
        if metrics["is_dag"]:
            try:
                metrics["longest_path_len"] = len(nx.dag_longest_path(actual_pred))
            except nx.NetworkXNoPath:
                metrics["longest_path_len"] = 0
        else:
            metrics["longest_path_len"] = None

        # Gold-based stats or relative metrics
        if gold is not None:
            gold_nodes = set(gold.nodes())
            pred_nodes = set(pred.nodes())
            node_intersection = gold_nodes & pred_nodes
            metrics["node_recall"] = (
                len(node_intersection) / len(gold_nodes) if gold_nodes else 1.0
            )

            gold_edges = MetricsCalculator._edge_set(gold)
            pred_edges = MetricsCalculator._edge_set(pred)
            edge_intersection = gold_edges & pred_edges

            precision = len(edge_intersection) / len(pred_edges) if pred_edges else 1.0
            recall = len(edge_intersection) / len(gold_edges) if gold_edges else 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

            metrics.update({
                "edge_precision": precision,
                "edge_recall": recall,
                "edge_f1": f1,
            })
        else:
            # When no gold standard, provide relative quality metrics
            # These will be meaningful for comparison between approaches
            metrics.update({
                "edge_precision": None,
                "edge_recall": None, 
                "edge_f1": None,
                "node_recall": None,
                # Add relative structural quality metrics
                "connectivity_ratio": metrics["num_edges"] / max(1, metrics["num_nodes"] - 1) if metrics["num_nodes"] > 1 else 0,
                "structure_quality": min(1.0, metrics["density"] * 2) if metrics["density"] else 0  # Normalized density score
            })

        return metrics
    
    @staticmethod
    def execution_metrics(dag: nx.DiGraph, durations: Dict[str, float], 
                         resources: Optional[Dict[str, Any]] = None, *, 
                         exclude_group_nodes: bool = True) -> Dict[str, Any]:
        """Calculate execution-oriented quality measures."""
        # Filter out group nodes for consistent critical path calculation if requested
        if exclude_group_nodes:
            actual_nodes = [n for n in dag.nodes() if not dag.nodes[n].get('is_group', False)]
            filtered_dag = dag.subgraph(actual_nodes).copy()
            filtered_durations = {n: durations.get(n, 1.0) for n in actual_nodes}
        else:
            filtered_dag = dag
            filtered_durations = durations
            
        serial_time = sum(filtered_durations.get(n, 1.0) for n in filtered_dag.nodes())

        try:
            crit_time = MetricsCalculator._critical_path_time(filtered_dag, filtered_durations)
        except ValueError:
            crit_time = None

        parallel_eff = (
            1.0 - crit_time / serial_time if crit_time and serial_time else 0.0
        )

        # Resource conflict rate
        conflict_edges = 0
        if resources is not None and filtered_dag.number_of_edges() > 0:
            for u, v in filtered_dag.edges():
                r_u, r_v = resources.get(u), resources.get(v)
                if r_u is None or r_v is None:
                    continue
                if isinstance(r_u, set) and isinstance(r_v, set):
                    if r_u & r_v:
                        conflict_edges += 1
                elif r_u == r_v:
                    conflict_edges += 1
            conflict_rate = conflict_edges / filtered_dag.number_of_edges()
        else:
            conflict_rate = 0.0

        return {
            "serial_time": serial_time,
            "critical_path_time": crit_time,
            # ── 새 필드들 ────────────────────
            "parallel_efficiency_path": parallel_eff,
            "parallel_efficiency": parallel_eff,               # (이전 호환용)
            # 필요 시 블록 기반 값도 넣을 수 있도록 placeholder
            "parallel_efficiency_block": None,                 # 채워 넣을 곳 마련
            "resource_conflict_rate": conflict_rate,
        }
    
    @staticmethod
    def complexity_metrics(workflow_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate complexity metrics for a workflow."""
        subtasks = workflow_data.get('subtasks', [])
        dag = workflow_data.get('dag')
        
        if not dag or not subtasks:
            return {
                "domain_complexity": 5.0,
                "coordination_complexity": 5.0,
                "computational_complexity": 5.0,
                "mode_heterogeneity": 0.5,
                "structural_complexity": 0.5
            }
        
        # Mode heterogeneity
        wide_count = len([st for st in subtasks if getattr(st, 'mode', None) == 'WIDE'])
        deep_count = len([st for st in subtasks if getattr(st, 'mode', None) == 'DEEP'])
        total_count = len(subtasks)
        
        if total_count > 0:
            wide_ratio = wide_count / total_count
            mode_heterogeneity = 2 * wide_ratio * (1 - wide_ratio)  # Max at 0.5/0.5 split
        else:
            mode_heterogeneity = 0.0
        
        # Structural complexity based on DAG properties
        if dag.number_of_nodes() > 0:
            density = nx.density(dag)
            avg_degree = sum(dict(dag.degree()).values()) / dag.number_of_nodes()
            structural_complexity = min(1.0, (density + avg_degree / dag.number_of_nodes()) / 2)
        else:
            structural_complexity = 0.0
        
        return {
            "domain_complexity": min(10.0, total_count * 0.5 + 3.0),  # Rough estimate
            "coordination_complexity": min(10.0, dag.number_of_edges() * 0.1 + 2.0),
            "computational_complexity": min(10.0, wide_count * 0.8 + deep_count * 1.2 + 2.0),
            "mode_heterogeneity": mode_heterogeneity,
            "structural_complexity": structural_complexity
        }
    
    @staticmethod
    def _edge_set(g: nx.DiGraph) -> Set[Tuple[str, str]]:
        """Return set of directed edges as (u, v) tuples."""
        return {(u, v) for u, v in g.edges()}
    
    @staticmethod
    def _critical_path_time(dag: nx.DiGraph, durations: Dict[str, float]) -> float:
        """Calculate longest path cost according to durations."""
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Critical path defined only for DAGs.")

        dist: Dict[str, float] = {}
        for node in nx.topological_sort(dag):
            preds = dag.predecessors(node)
            dist[node] = max((dist[p] for p in preds), default=0.0) + durations.get(node, 1.0)
        return max(dist.values(), default=0.0)


class QualityAssessor:
    """Unified quality assessment for workflows."""
    
    @staticmethod
    def assess_workflow_quality(workflow_data: Dict[str, Any], 
                               subtasks: List[SubTask],
                               dag: nx.DiGraph) -> Dict[str, float]:
        """Assess overall workflow quality."""
        
        # Basic completeness check
        if not subtasks:
            completeness = 0.0
        else:
            completeness = min(1.0, len(subtasks) / 5.0)  # Assume 5 subtasks = complete
        
        # Coherence based on DAG structure
        if dag.number_of_nodes() == 0:
            coherence = 0.0
        elif not nx.is_directed_acyclic_graph(dag):
            coherence = 0.0
        else:
            # Base coherence on connectivity
            expected_edges = len(subtasks) * 0.3  # Expect ~30% connectivity
            actual_edges = dag.number_of_edges()
            coherence = min(1.0, actual_edges / max(expected_edges, 1))
        
        # Efficiency based on parallel structure
        try:
            if dag.number_of_nodes() > 0:
                levels = {}
                for node in nx.topological_sort(dag):
                    preds = list(dag.predecessors(node))
                    levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
                
                max_level = max(levels.values()) if levels else 0
                efficiency = 1.0 - (max_level + 1) / dag.number_of_nodes()
                efficiency = max(0.0, efficiency)
            else:
                efficiency = 0.0
        except:
            efficiency = 0.5
        
        # Feasibility - assume high unless there are obvious problems
        feasibility = 0.8
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(dag))
        if isolated_nodes:
            efficiency *= 0.9
            feasibility *= 0.9
        
        # Overall quality weighted average
        weights = [0.3, 0.25, 0.25, 0.2]  # completeness, coherence, efficiency, feasibility
        overall_quality = sum(w * s for w, s in zip(weights, [completeness, coherence, efficiency, feasibility]))
        
        return {
            "completeness_score": round(completeness, 3),
            "coherence_score": round(coherence, 3),
            "efficiency_score": round(efficiency, 3),
            "feasibility_score": round(feasibility, 3),
            "overall_quality": round(overall_quality, 3)
        }
    
    @staticmethod
    def generate_recommendations(per_approach_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommendations based on evaluation results."""
        
        if not per_approach_metrics:
            return {"error": "No approaches to evaluate"}
        
        # Normalize build_time for comparison (invert so lower time = higher score)
        build_times = [metrics.get("build_time", 1) for metrics in per_approach_metrics.values()]
        max_build_time = max(build_times) if build_times else 1
        
        normalized_metrics = {}
        for approach, metrics in per_approach_metrics.items():
            normalized = metrics.copy()
            # Invert build_time so faster approaches get higher scores
            normalized["build_speed"] = 1 - (metrics.get("build_time", 1) / max_build_time)
            normalized_metrics[approach] = normalized
        
        # Scoring weights for different use cases
        use_case_weights = {
            "speed_focused": {
                "build_speed": 0.4,  # Use build_speed instead of build_time
                "parallel_efficiency": 0.3,
                "edge_f1": 0.1,
                "node_recall": 0.1,
                "connectivity_ratio": 0.05,
                "structure_quality": 0.05,
                "resource_conflict_rate": -0.1,
            },
            "quality_focused": {
                "edge_f1": 0.3,
                "node_recall": 0.25,
                "connectivity_ratio": 0.15,
                "structure_quality": 0.15,
                "parallel_efficiency": 0.1,
                "resource_conflict_rate": -0.1,
            },
            "balanced": {
                "build_speed": 0.2,
                "parallel_efficiency": 0.2,
                "edge_f1": 0.2,
                "node_recall": 0.15,
                "connectivity_ratio": 0.125,
                "structure_quality": 0.125,
                "resource_conflict_rate": -0.1,
            }
        }
        
        recommendations = {}
        
        for use_case, weights in use_case_weights.items():
            best_approach = None
            best_score = float('-inf')
            
            scores = {}
            for approach, metrics in normalized_metrics.items():
                score = 0
                for metric, weight in weights.items():
                    value = metrics.get(metric, 0)
                    # Skip None values (missing gold standard metrics) but count relative metrics
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
                                                               normalized_metrics[best_approach] if best_approach else {})
            }
        
        return recommendations
    
    @staticmethod
    def _generate_reasoning(approach: str, use_case: str, metrics: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for recommendation."""
        
        if not approach:
            return "No suitable approach found"
        
        reasoning_parts = [f"{approach} is recommended for {use_case} use case because:"]
        
        if use_case == "speed_focused":
            build_speed = metrics.get("build_speed", 0)
            parallel_eff = metrics.get("parallel_efficiency", 0)
            build_time = metrics.get("build_time", 0)
            reasoning_parts.append(f"- Fast build time ({build_time:.2f}s)")
            reasoning_parts.append(f"- Good parallel efficiency ({parallel_eff:.3f})")
            
        elif use_case == "quality_focused":
            edge_f1 = metrics.get("edge_f1")
            node_recall = metrics.get("node_recall")
            connectivity = metrics.get("connectivity_ratio", 0)
            structure_quality = metrics.get("structure_quality", 0)
            
            if edge_f1 is not None and node_recall is not None:
                reasoning_parts.append(f"- Excellent edge accuracy (F1: {edge_f1:.3f})")
                reasoning_parts.append(f"- High node recall ({node_recall:.3f})")
            else:
                reasoning_parts.append(f"- Good structural connectivity ({connectivity:.3f})")
                reasoning_parts.append(f"- High structure quality ({structure_quality:.3f})")
            
        elif use_case == "balanced":
            reasoning_parts.append("- Well-balanced performance across all metrics")
        
        conflict_rate = metrics.get("resource_conflict_rate", 0)
        if conflict_rate < 0.1:
            reasoning_parts.append(f"- Low resource conflicts ({conflict_rate:.3f})")
        
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
        """Evaluate results from compare_dag_approaches()."""
        
        # Extract task_id from successful approach
        task_id = None
        for approach_name, result in comparison_results.items():
            if approach_name != "comparison_summary" and result.get("success", False):
                task_id = result.get("metrics", {}).get("task_id")
                break
        
        # Get ground truth data
        gold = self.gold_dags.get(task_id) if task_id else None
        durations = self.durations_db.get(task_id, {}) if task_id else {}
        resources = self.resources_db.get(task_id, {}) if task_id else None
        
        # Evaluate each approach
        per_approach = {}
        for approach_name, result in comparison_results.items():
            if approach_name == "comparison_summary" or not result.get("success", False):
                continue
            
            dag: nx.DiGraph = result["dag"]
            
            structural = MetricsCalculator.structural_metrics(dag, gold)
            execution = MetricsCalculator.execution_metrics(dag, durations, resources)
            
            # Add build performance metrics for recommendations
            build_metrics = result.get("metrics", {})
            performance_metrics = {
                "build_time": build_metrics.get("build_time", 0),
                "total_llm_calls": build_metrics.get("total_llm_calls", 0)
            }
            
            per_approach[approach_name] = {**structural, **execution, **performance_metrics}
        
        # Generate aggregate statistics
        aggregate_summary = {}
        if per_approach:
            keys = next(iter(per_approach.values())).keys()
            for k in keys:
                vals = [v[k] for v in per_approach.values() if v[k] is not None]
                if not vals:
                    continue
                aggregate_summary[k] = {
                    "mean": statistics.mean(vals),
                    "best": max(vals) if k not in {"critical_path_time", "resource_conflict_rate", "serial_time"} else min(vals),
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
