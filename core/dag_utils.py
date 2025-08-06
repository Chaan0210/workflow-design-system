# core/dag_utils.py
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import networkx as nx

from models import SubTask


class DAGProcessor:
    """Unified DAG processing and manipulation utilities."""
    
    @staticmethod
    def apply_rule_based_pruning(dag: nx.DiGraph) -> nx.DiGraph:
        """Apply rule-based pruning for unit conversion/rounding/format tasks."""
        
        # Keywords that indicate unit conversion, rounding, or formatting tasks
        FORMATTING_KEYWORDS = [
            'unit', 'convert', 'conversion', 'round', 'rounding', 'format', 'formatting',
            'display', 'present', 'show', 'output', 'print', 'render', 'style',
            'decimal', 'precision', 'scale', 'normalize', 'standardize'
        ]
        
        # Find formatting/conversion tasks
        formatting_tasks = set()
        for node_id in dag.nodes():
            node_data = dag.nodes[node_id]
            description = node_data.get('description', '').lower()
            
            # Check if this is a formatting/conversion task
            if any(keyword in description for keyword in FORMATTING_KEYWORDS):
                formatting_tasks.add(node_id)
        
        # For each formatting task, ensure it has at most one direct predecessor
        edges_to_remove = []
        for format_task in formatting_tasks:
            predecessors = list(dag.predecessors(format_task))
            
            if len(predecessors) > 1:
                # Keep only the most confident predecessor, remove others
                best_pred = None
                best_confidence = -1
                
                for pred in predecessors:
                    edge_data = dag[pred][format_task]
                    confidence = edge_data.get('confidence', 0.5)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_pred = pred
                
                # Remove all edges except the best one
                for pred in predecessors:
                    if pred != best_pred:
                        edges_to_remove.append((pred, format_task))
        
        # Remove the identified edges
        for u, v in edges_to_remove:
            if dag.has_edge(u, v):
                dag.remove_edge(u, v)
        
        return dag
    
    @staticmethod
    def resolve_cycles_intelligently(dag: nx.DiGraph, edge_conf: Dict[Tuple[str, str], float]) -> nx.DiGraph:
        """Remove cycles by eliminating lowest confidence edges."""
        try:
            while True:
                cycles = list(nx.find_cycle(dag, orientation="original"))
                if not cycles:
                    break

                weakest = None
                min_c = float('inf')
                for u, v, _ in cycles:
                    c = edge_conf.get((u, v), 0.5)
                    if c < min_c:
                        min_c = c
                        weakest = (u, v)
                if weakest:
                    dag.remove_edge(*weakest)
        except nx.NetworkXNoCycle:
            pass
        return dag
    
    @staticmethod
    def post_process_graph(dag: nx.DiGraph) -> None:
        """Apply consistent post-processing to DAG."""
        # 1) Confidence rounding/format
        for u, v, d in dag.edges(data=True):
            if "confidence" in d:
                d["confidence"] = round(float(d["confidence"]), 3)

        # 2) Ensure critical path, parallel levels
        try:
            critical_path = nx.algorithms.dag.dag_longest_path(dag)
        except (nx.NetworkXUnfeasible, nx.NetworkXError):
            critical_path = []

        for n in dag.nodes():
            dag.nodes[n]['on_critical_path'] = False
            dag.nodes[n]['critical_path_position'] = None

        for i, n in enumerate(critical_path):
            dag.nodes[n]['on_critical_path'] = True
            dag.nodes[n]['critical_path_position'] = i

        # 3) Parallel levels
        try:
            levels = {}
            for node in nx.topological_sort(dag):
                preds = list(dag.predecessors(node))
                levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1

            parallel_blocks = defaultdict(list)
            for node, level in levels.items():
                parallel_blocks[level].append(node)

            for level, nodes in parallel_blocks.items():
                for node in nodes:
                    dag.nodes[node]['parallel_level'] = level
                    dag.nodes[node]['parallel_peers'] = [n for n in nodes if n != node]
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph still contains cycles after post-process().")
    
    @staticmethod
    def calculate_structural_metrics(dag: nx.DiGraph) -> Dict[str, float]:
        """Calculate structural metrics for a DAG."""
        if dag.number_of_nodes() == 0:
            return {
                "density": 0.0,
                "average_degree": 0.0,
                "longest_path_length": 0,
                "parallel_efficiency": 0.0
            }
        
        density = nx.density(dag)
        avg_degree = sum(dict(dag.degree()).values()) / dag.number_of_nodes()
        
        try:
            longest_path = nx.dag_longest_path(dag) if nx.is_directed_acyclic_graph(dag) else []
            longest_path_length = len(longest_path)
        except:
            longest_path_length = 0
            
        # Calculate parallel efficiency
        if dag.number_of_nodes() > 0:
            try:
                levels = {}
                for node in nx.topological_sort(dag):
                    preds = list(dag.predecessors(node))
                    levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
                
                max_level = max(levels.values()) if levels else 0
                parallel_efficiency = 1.0 - (max_level + 1) / dag.number_of_nodes()
            except:
                parallel_efficiency = 0.0
        else:
            parallel_efficiency = 0.0
        
        return {
            "density": density,
            "average_degree": avg_degree,
            "longest_path_length": longest_path_length,
            "parallel_efficiency": max(0.0, parallel_efficiency)
        }
    
    @staticmethod
    def validate_dag_structure(dag: nx.DiGraph) -> Dict[str, any]:
        """Validate DAG structure and return issues."""
        issues = {
            "is_valid_dag": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if it's a DAG
        if not nx.is_directed_acyclic_graph(dag):
            issues["is_valid_dag"] = False
            issues["errors"].append("Graph contains cycles")
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(dag))
        if isolated_nodes:
            issues["warnings"].append(f"Isolated nodes found: {isolated_nodes}")
        
        # Check for nodes without obj attributes
        nodes_without_obj = [n for n in dag.nodes() if 'obj' not in dag.nodes[n]]
        if nodes_without_obj:
            issues["warnings"].append(f"Nodes without obj attribute: {nodes_without_obj}")
        
        return issues


# Convenience function for backward compatibility
def apply_rule_based_pruning(dag: nx.DiGraph) -> nx.DiGraph:
    """Apply rule-based pruning (convenience function)."""
    return DAGProcessor.apply_rule_based_pruning(dag)
