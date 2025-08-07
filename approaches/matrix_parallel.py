# approaches/matrix_parallel.py
from typing import Dict, Any, List, Tuple
import networkx as nx

from models import SubTask
from .base import DAGApproach
from prompts import PromptManager


class MatrixParallelApproach(DAGApproach):
    def __init__(self):
        super().__init__("matrix_parallel")
        self.fallback_threshold = 10  # Fallback to sequential for large matrices
    
    def _build_dag_impl(self, subtasks: List[SubTask], original_task: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """병렬성을 고려한 개선된 matrix DAG 구성"""
        
        print("⚡ MATRIX-PARALLEL APPROACH - Enhanced parallel matrix generation...")
        print(f"   Original task: {original_task[:100]}...")
        print(f"   Subtasks ({len(subtasks)}): {[st.id for st in subtasks]}")
        
        # Fallback for large matrices
        if len(subtasks) >= self.fallback_threshold:
            print(f"   🔄 Using fallback: too many subtasks ({len(subtasks)} >= {self.fallback_threshold})")
            dag = self._create_sequential_dag(subtasks)
            metrics = {
                "total_llm_calls": 0,
                "used_fallback": True,
                "fallback_reason": f"n={len(subtasks)} >= threshold={self.fallback_threshold}",
                "parallel_optimization": False
            }
            return dag, metrics
        
        # Build enhanced matrix with parallel optimization
        try:
            print("   📤 Sending enhanced LLM request for parallel-optimized matrix...")
            matrix_data = self._build_enhanced_matrix(subtasks, original_task)
            print("   📥 Enhanced matrix response received:")
            self._print_enhanced_matrix(matrix_data)
            
            print("   🔧 Converting enhanced matrix to DAG...")
            dag = self._enhanced_matrix_to_dag(matrix_data, subtasks)
            print(f"   ✅ Enhanced DAG created: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
            print(f"   📊 DAG edges: {list(dag.edges())}")
            
            # Calculate parallel efficiency from blocks
            block_efficiency = self._calculate_parallel_efficiency_from_blocks(
                matrix_data.get("parallel_blocks", [])
            )
            print(f"   ⚡ Block parallel efficiency: {block_efficiency:.3f}")
            
            llm_calls = 1
            fallback_used = False
            
        except Exception as e:
            print(f"   ❌ Enhanced matrix approach failed: {e}")
            print("   🔄 Using sequential fallback...")
            # Fallback to sequential
            dag = self._create_sequential_dag(subtasks)
            llm_calls = 0
            fallback_used = True
            block_efficiency = 0.0
        
        # Calculate path-based efficiency for comparison
        path_efficiency = self._calculate_path_based_efficiency(dag) if not fallback_used else 0.0
        
        metrics = {
            "llm_matrix_calls": llm_calls,
            "total_llm_calls": llm_calls,
            "matrix_size": len(subtasks) ** 2,
            "used_fallback": fallback_used,
            "parallel_efficiency_block": block_efficiency,  # From parallel blocks analysis
            "parallel_efficiency_path": path_efficiency,    # From critical path analysis
            "parallel_efficiency": path_efficiency,         # Keep for backward compatibility
            "parallel_optimization": True,
            "approach_specific": "enhanced_parallel_matrix"
        }
        
        return dag, metrics
    
    def _build_enhanced_matrix(self, subtasks: List[SubTask], original_task: str) -> Dict[str, Any]:
        """개선된 병렬 최적화 매트릭스 생성"""
        subtask_list = "\n".join([f"{st.id}: {st.description}" for st in subtasks])
        prompt = PromptManager.format_enhanced_matrix_generation_prompt(original_task, subtask_list)
        
        return self.llm_client.call_json(prompt, validator=self._validate_enhanced_matrix_json)
    
    def _enhanced_matrix_to_dag(self, matrix_data: Dict[str, Any], subtasks: List[SubTask]) -> nx.DiGraph:
        """개선된 매트릭스를 DAG로 변환"""
        dag = nx.DiGraph()
        
        # Add nodes
        for st in subtasks:
            dag.add_node(st.id, obj=st, description=st.description)
        
        # Add edges from matrix with confidence cutoff
        matrix = matrix_data["matrix"]
        conf_matrix = matrix_data["confidence_matrix"]
        task_order = matrix_data["task_order"]
        parallel_blocks = matrix_data.get("parallel_blocks", [])
        
        print(f"      Detected parallel blocks: {parallel_blocks}")
        print(f"      Applying confidence cutoff (≥0.3)...")
        
        edges_added = 0
        edges_rejected = 0
        
        for i, dependent_task in enumerate(task_order):
            for j, prerequisite_task in enumerate(task_order):
                if matrix[i][j] == 1:  # dependent_task depends on prerequisite_task
                    confidence = conf_matrix[i][j]
                    if confidence >= 0.3:
                        dag.add_edge(prerequisite_task, dependent_task,
                                   confidence=confidence,
                                   matrix_edge=True,
                                   parallel_optimized=True)
                        edges_added += 1
                        print(f"         ✅ Added edge: {prerequisite_task} → {dependent_task} (conf: {confidence})")
                    else:
                        edges_rejected += 1
                        print(f"         ❌ Rejected edge: {prerequisite_task} → {dependent_task} (conf: {confidence} < 0.3)")
        
        print(f"      Summary: {edges_added} edges added, {edges_rejected} edges rejected")
        
        # 병렬 블록 정보를 노드에 메타데이터로 저장
        for i, block in enumerate(parallel_blocks):
            for task_id in block:
                if task_id in dag.nodes:
                    dag.nodes[task_id]['parallel_block'] = i
        
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
        """사이클 해결 (기존 matrix.py와 동일)"""
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
                    # u is prerequisite, v is dependent, so matrix[v][u] = 1 means v depends on u
                    v_idx, u_idx = id_to_idx[v], id_to_idx[u]
                    conf = conf_matrix[v_idx][u_idx]
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
    
    def _print_enhanced_matrix(self, matrix_data: Dict[str, Any]):
        """개선된 매트릭스 데이터 출력"""
        matrix = matrix_data["matrix"]
        conf_matrix = matrix_data["confidence_matrix"]
        task_order = matrix_data["task_order"]
        parallel_blocks = matrix_data.get("parallel_blocks", [])
        
        print(f"      Task order: {task_order}")
        print(f"      Parallel blocks: {parallel_blocks}")
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
    
    def _calculate_parallel_efficiency_from_blocks(self, parallel_blocks: List[List[str]]) -> float:
        """병렬 블록을 기반으로 병렬 효율성 계산"""
        if not parallel_blocks:
            return 0.0
        
        total_tasks = sum(len(block) for block in parallel_blocks)
        if total_tasks == 0:
            return 0.0
        
        # 각 블록에서 병렬 실행 가능한 작업의 비율 계산
        parallel_tasks = sum(len(block) for block in parallel_blocks if len(block) > 1)
        
        return parallel_tasks / total_tasks
    
    def _calculate_path_based_efficiency(self, dag: nx.DiGraph) -> float:
        """Calculate parallel efficiency based on critical path analysis."""
        if dag.number_of_nodes() <= 1:
            return 1.0
        
        try:
            # Calculate level-based parallelization
            levels = {}
            for node in nx.topological_sort(dag):
                preds = list(dag.predecessors(node))
                levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
            
            max_level = max(levels.values()) if levels else 0
            return max(0.0, 1.0 - (max_level + 1) / dag.number_of_nodes())
        except:
            return 0.5
    
    def _validate_enhanced_matrix_json(self, d: dict) -> None:
        """개선된 매트릭스 JSON 구조 검증"""
        # 기본 매트릭스 검증
        if not isinstance(d, dict):
            raise ValueError(f"Expected dict, got {type(d)}")
            
        if not isinstance(d.get("matrix"), list):
            raise ValueError(f"Missing or invalid 'matrix' field: {d.get('matrix')}")
            
        if not isinstance(d.get("confidence_matrix"), list):
            raise ValueError(f"Missing or invalid 'confidence_matrix' field: {d.get('confidence_matrix')}")
            
        if not isinstance(d.get("task_order"), list):
            raise ValueError(f"Missing or invalid 'task_order' field: {d.get('task_order')}")
        
        # 병렬 블록 검증 (선택적)
        if "parallel_blocks" in d:
            if not isinstance(d["parallel_blocks"], list):
                raise ValueError(f"Invalid 'parallel_blocks' field: {d.get('parallel_blocks')}")
            
            for block in d["parallel_blocks"]:
                if not isinstance(block, list):
                    raise ValueError(f"Each parallel block must be a list: {block}")
        
        # 기본 매트릭스 차원 검증
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