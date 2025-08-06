# approaches/base.py
import time
from typing import Dict, Any, List, Tuple
import networkx as nx

from models import SubTask
from core import LLMClient, DAGProcessor


class DAGApproach:
    """Base class for DAG building approaches."""
    
    def __init__(self, name: str):
        self.name = name
        self.llm_client = LLMClient()
        self.dag_processor = DAGProcessor()
    
    def build_dag(self, subtasks: List[SubTask], original_task: str = "") -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """Build DAG and return metrics."""
        start_time = time.time()
        
        try:
            dag, approach_metrics = self._build_dag_impl(subtasks, original_task)
            build_time = time.time() - start_time
            
            # Standard metrics
            metrics = {
                "approach": self.name,
                "build_time": build_time,
                "nodes": dag.number_of_nodes(),
                "edges": dag.number_of_edges(),
                "success": True,
                **approach_metrics
            }
            
            return dag, metrics
            
        except Exception as e:
            build_time = time.time() - start_time
            metrics = {
                "approach": self.name,
                "build_time": build_time,
                "success": False,
                "error": str(e),
                "nodes": 0,
                "edges": 0
            }
            
            # Return empty DAG on failure
            empty_dag = nx.DiGraph()
            return empty_dag, metrics
    
    def _build_dag_impl(self, subtasks: List[SubTask], original_task: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """Override this method in subclasses."""
        raise NotImplementedError
    
    def _create_sequential_dag(self, subtasks: List[SubTask]) -> nx.DiGraph:
        """Create sequential DAG as fallback."""
        dag = nx.DiGraph()
        
        for st in subtasks:
            dag.add_node(st.id, obj=st, description=st.description)
        
        # Sequential chain
        for i in range(len(subtasks) - 1):
            dag.add_edge(subtasks[i].id, subtasks[i + 1].id, confidence=0.8)
        
        self.dag_processor.post_process_graph(dag)
        return dag
