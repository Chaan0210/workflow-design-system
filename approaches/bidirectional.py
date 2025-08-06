# approaches/bidirectional.py
import asyncio
from typing import Dict, Any, List, Tuple
import networkx as nx
import itertools

from models import SubTask
from workflow_engine import WorkflowEngine
from .base import DAGApproach


class BidirectionalApproach(DAGApproach):
    def __init__(self):
        super().__init__("bidirectional")
        self.workflow_engine = WorkflowEngine()
    
    def _build_dag_impl(self, subtasks: List[SubTask], original_task: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """Use the workflow engine's DAG building logic."""
        
        print("🔄 BIDIRECTIONAL APPROACH - Using WorkflowEngine DAG building...")
        print(f"   Original task: {original_task[:100]}...")
        print(f"   Subtasks ({len(subtasks)}): {[st.id for st in subtasks]}")
        
        # Calculate expected LLM calls before building
        n_pairs = len(subtasks) * (len(subtasks) - 1) // 2
        dependency_calls = n_pairs * 2  # bidirectional
        print(f"   📊 Expected dependency LLM calls: {dependency_calls} ({n_pairs} pairs × 2 directions)")
        
        print("   📤 Delegating to WorkflowEngine._build_dag()...")
        dag = self.workflow_engine._build_dag(subtasks, original_task)
        
        # Calculate actual metrics
        resource_calls = dag.number_of_edges()  # only for actual edges
        total_calls = dependency_calls + resource_calls
        
        print(f"   📥 DAG received from WorkflowEngine:")
        print(f"      - Nodes: {dag.number_of_nodes()}")
        print(f"      - Edges: {dag.number_of_edges()}")
        print(f"      - Node IDs: {list(dag.nodes())}")
        print(f"      - Edge pairs: {list(dag.edges())}")
        print(f"   📊 Resource LLM calls: {resource_calls} (1 per actual edge)")
        print(f"   📊 Total LLM calls: {total_calls}")
        
        metrics = {
            "llm_dependency_calls": dependency_calls,
            "llm_resource_calls": resource_calls,
            "total_llm_calls": total_calls,
        }
        
        return dag, metrics
