# approaches/matrix.py
from typing import Dict, Any, List, Tuple
import networkx as nx

from models import SubTask
from .base import DAGApproach
from prompts import PromptManager


class MatrixApproach(DAGApproach):
    """
    Matrix DAG building approach.
    
    Uses adjacency matrix generation with confidence cutoff:
    1. Generate dependency adjacency matrix via LLM
    2. Apply confidence cutoff (≥0.5) to accept edges
    3. Remove cycles by eliminating lowest confidence edges
    4. Fallback to sequential for large matrices (≥10 tasks)
    """
    
    def __init__(self):
        super().__init__("matrix")
        self.fallback_threshold = 10  # Fallback to sequential for large matrices
    
    def _build_dag_impl(self, subtasks: List[SubTask], original_task: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """Build DAG using adjacency matrix approach."""
        
        print("🔢 MATRIX APPROACH - Starting adjacency matrix generation...")
        print(f"   Original task: {original_task[:100]}...")
        print(f"   Subtasks ({len(subtasks)}): {[st.id for st in subtasks]}")
        
        # Fallback for large matrices
        if len(subtasks) >= self.fallback_threshold:
            print(f"   🔄 Using fallback: too many subtasks ({len(subtasks)} >= {self.fallback_threshold})")
            dag = self._create_sequential_dag(subtasks)
            metrics = {
                "total_llm_calls": 0,
                "used_fallback": True,
                "fallback_reason": f"n={len(subtasks)} >= threshold={self.fallback_threshold}"
            }
            return dag, metrics
        
        # Build matrix
        try:
            print("   📤 Sending LLM request for adjacency matrix...")
            matrix_data = self._build_matrix(subtasks, original_task)
            print("   📥 LLM Response received. Matrix data:")
            self._print_matrix(matrix_data)
            
            print("   🔧 Converting matrix to DAG...")
            dag = self._matrix_to_dag(matrix_data, subtasks)
            print(f"   ✅ DAG created: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
            print(f"   📊 DAG edges: {list(dag.edges())}")
            
            llm_calls = 1
            fallback_used = False
            
        except Exception as e:
            print(f"   ❌ Matrix approach failed: {e}")
            print("   🔄 Using sequential fallback...")
            # Fallback to sequential
            dag = self._create_sequential_dag(subtasks)
            llm_calls = 0
            fallback_used = True
        
        metrics = {
            "llm_matrix_calls": llm_calls,
            "total_llm_calls": llm_calls,
            "matrix_size": len(subtasks) ** 2,
            "used_fallback": fallback_used
        }
        
        return dag, metrics
    
    def _print_matrix(self, matrix_data: Dict[str, Any]):
        """Print matrix data for debugging."""
        matrix = matrix_data["matrix"]
        conf_matrix = matrix_data["confidence_matrix"]
        task_order = matrix_data["task_order"]
        
        print(f"      Task order: {task_order}")
        print(f"      Dependency matrix ({len(matrix)}x{len(matrix[0]) if matrix else 0}):")
        
        # Print header
        header = "        " + " ".join(f"{task:>3}" for task in task_order)
        print(header)
        
        # Print dependency matrix rows
        for i, row in enumerate(matrix):
            row_str = f"     {task_order[i]:>3} " + " ".join(f"{val:>3}" for val in row)
            print(row_str)
        
        # Print confidence matrix
        print(f"      Confidence matrix:")
        print(header)
        for i, row in enumerate(conf_matrix):
            row_str = f"     {task_order[i]:>3} " + " ".join(f"{val:>3.1f}" for val in row)
            print(row_str)
    
    def _build_matrix(self, subtasks: List[SubTask], original_task: str) -> Dict[str, Any]:
        """Build adjacency matrix."""
        subtask_list = "\n".join([f"{st.id}: {st.description}" for st in subtasks])
        prompt = PromptManager.format_matrix_generation_prompt(original_task, subtask_list)
        
        return self.llm_client.call_json(prompt, validator=self._validate_matrix_json)
    
    def _matrix_to_dag(self, matrix_data: Dict[str, Any], subtasks: List[SubTask]) -> nx.DiGraph:
        """Convert matrix to DAG."""
        dag = nx.DiGraph()
        
        # Add nodes
        for st in subtasks:
            dag.add_node(st.id, obj=st, description=st.description)
        
        # Add edges from matrix with confidence cutoff
        matrix = matrix_data["matrix"]
        conf_matrix = matrix_data["confidence_matrix"]
        task_order = matrix_data["task_order"]
        
        print(f"      Applying confidence cutoff (≥0.5)...")
        edges_added = 0
        edges_rejected = 0
        
        for i, from_task in enumerate(task_order):
            for j, to_task in enumerate(task_order):
                if matrix[i][j] == 1:
                    confidence = conf_matrix[i][j]
                    if confidence >= 0.5:
                        dag.add_edge(from_task, to_task,
                                   confidence=confidence,
                                   matrix_edge=True)
                        edges_added += 1
                        print(f"         ✅ Added edge: {from_task} → {to_task} (conf: {confidence})")
                    else:
                        edges_rejected += 1
                        print(f"         ❌ Rejected edge: {from_task} → {to_task} (conf: {confidence} < 0.5)")
        
        print(f"      Summary: {edges_added} edges added, {edges_rejected} edges rejected")
        
        # Remove cycles if any
        if not nx.is_directed_acyclic_graph(dag):
            print(f"      🔄 Cycles detected, resolving...")
            dag = self._resolve_cycles(dag, conf_matrix, task_order)
        else:
            print(f"      ✅ No cycles detected")
        
        # Post-process
        self.dag_processor.post_process_graph(dag)
        dag = self.dag_processor.apply_rule_based_pruning(dag)
        
        return dag
    
    def _resolve_cycles(self, dag: nx.DiGraph, conf_matrix: List[List[float]], task_order: List[str]) -> nx.DiGraph:
        """Remove cycles by eliminating lowest confidence edges."""
        id_to_idx = {task_id: i for i, task_id in enumerate(task_order)}
        cycles_resolved = 0
        
        while not nx.is_directed_acyclic_graph(dag):
            try:
                cycle = nx.find_cycle(dag, orientation="original")
                print(f"         Found cycle: {[edge[:2] for edge in cycle]}")
                
                # Find weakest edge in cycle
                weakest_edge = None
                min_conf = float('inf')
                
                for u, v, _ in cycle:
                    i, j = id_to_idx[u], id_to_idx[v]
                    conf = conf_matrix[i][j]
                    if conf < min_conf:
                        min_conf = conf
                        weakest_edge = (u, v)
                
                if weakest_edge:
                    print(f"         Removing weakest edge: {weakest_edge} (conf: {min_conf})")
                    dag.remove_edge(*weakest_edge)
                    cycles_resolved += 1
                else:
                    # Fallback - shouldn't happen but just in case
                    first_edge = cycle[0][:2]
                    dag.remove_edge(*first_edge)
                    print(f"         Fallback: removed first edge {first_edge}")
                    cycles_resolved += 1
                    
            except nx.NetworkXNoCycle:
                break
        
        print(f"      ✅ Cycles resolved: {cycles_resolved}")
        return dag
    
    def _validate_matrix_json(self, d: dict) -> None:
        """Validate matrix JSON structure."""
        if not isinstance(d, dict):
            raise ValueError(f"Expected dict, got {type(d)}")
            
        if not isinstance(d.get("matrix"), list):
            raise ValueError(f"Missing or invalid 'matrix' field: {d.get('matrix')}")
            
        if not isinstance(d.get("confidence_matrix"), list):
            raise ValueError(f"Missing or invalid 'confidence_matrix' field: {d.get('confidence_matrix')}")
            
        if not isinstance(d.get("task_order"), list):
            raise ValueError(f"Missing or invalid 'task_order' field: {d.get('task_order')}")
        
        matrix = d["matrix"]
        conf_matrix = d["confidence_matrix"]
        n = len(matrix)
        
        if len(conf_matrix) != n:
            raise ValueError(f"Matrix dimensions mismatch: matrix={n}x?, conf_matrix={len(conf_matrix)}x?")
            
        if len(d["task_order"]) != n:
            raise ValueError(f"Task order length mismatch: expected {n}, got {len(d['task_order'])}")
        
        for i, row in enumerate(matrix):
            if not isinstance(row, list):
                raise ValueError(f"Matrix row {i} is not a list: {row}")
            if len(row) != n:
                raise ValueError(f"Matrix row {i} has wrong length: expected {n}, got {len(row)}")
            if len(conf_matrix[i]) != n:
                raise ValueError(f"Confidence matrix row {i} has wrong length: expected {n}, got {len(conf_matrix[i])}")
            if matrix[i][i] != 0:
                raise ValueError(f"Matrix diagonal at [{i}][{i}] must be 0, got {matrix[i][i]}")
