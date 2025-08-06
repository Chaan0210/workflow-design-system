# approaches/bidirectional_parallel.py
import asyncio
from typing import Dict, Any, List, Tuple
import networkx as nx
import itertools
import time

from models import SubTask
from .base import DAGApproach
from core import LLMClient, DAGProcessor
from core.validation_utils import DAGValidator


class BidirectionalParallelApproach(DAGApproach):    
    def __init__(self):
        super().__init__("bidirectional_parallel")
        self.workflow_engine = None  # WorkflowEngine 대신 직접 구현
        self.LOW_CONF_THRESHOLD = 0.4
        self.CUTOFF_THRESHOLD = 0.5
        self.BATCH_SIZE = 8  # 병렬 처리 배치 크기
    
    def _build_dag_impl(self, subtasks: List[SubTask], original_task: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """병렬 처리를 사용한 bidirectional DAG 구성"""
        
        print("⚡ BIDIRECTIONAL-PARALLEL APPROACH - Using async batch processing...")
        print(f"   Original task: {original_task[:100]}...")
        print(f"   Subtasks ({len(subtasks)}): {[st.id for st in subtasks]}")
        
        # DAG 초기화
        dag = nx.DiGraph()
        for st in subtasks:
            dag.add_node(st.id, obj=st, description=st.description)
        
        # 의존성 쌍 생성 (bidirectional)
        dependency_pairs = []
        for a, b in itertools.combinations(subtasks, 2):
            dependency_pairs.append((a, b))  # A -> B
            dependency_pairs.append((b, a))  # B -> A
        
        n_pairs = len(dependency_pairs)
        print(f"   📊 Total dependency checks: {n_pairs} (bidirectional pairs)")
        print(f"   📦 Batch size: {self.BATCH_SIZE} (processing {(n_pairs-1)//self.BATCH_SIZE + 1} batches)")
        
        # 비동기 병렬 의존성 분석
        start_time = time.time()
        dependency_results = asyncio.run(self.llm_client.batch_dependency_analysis(
            dependency_pairs, original_task, batch_size=self.BATCH_SIZE
        ))
        analysis_time = time.time() - start_time
        
        print(f"   ⚡ Parallel analysis completed in {analysis_time:.2f}s")
        print(f"   📊 Analysis rate: {len(dependency_pairs)/analysis_time:.1f} pairs/second")
        
        # DAG 구성
        edge_conf_map = {}
        low_conf_candidates = []
        resource_calls = 0
        
        for (task_a, task_b), (dependent, confidence) in dependency_results.items():
            if dependent and confidence >= self.CUTOFF_THRESHOLD:
                # 높은 신뢰도 의존성 추가
                if not dag.has_edge(task_b.id, task_a.id):  # 역방향 충돌 확인
                    dag.add_edge(task_a.id, task_b.id, confidence=confidence, 
                               has_resource_conflict=False, shared_resources=[])
                    edge_conf_map[(task_a.id, task_b.id)] = confidence
                    resource_calls += 1
            elif dependent and confidence >= self.LOW_CONF_THRESHOLD:
                # 낮은 신뢰도 후보
                low_conf_candidates.append((task_a.id, task_b.id, confidence))
        
        # 낮은 신뢰도 엣지 추가 (충돌 없는 경우만)
        for u, v, c in sorted(low_conf_candidates, key=lambda x: -x[2]):
            if not dag.has_edge(u, v) and not dag.has_edge(v, u):
                dag.add_edge(u, v, confidence=round(c, 3),
                           has_resource_conflict=False, shared_resources=[])
        
        print(f"   📊 Edges added: {dag.number_of_edges()} (high conf: {resource_calls}, low conf: {dag.number_of_edges()-resource_calls})")
        
        # 사이클 해결
        dag = self.dag_processor.resolve_cycles_intelligently(dag, edge_conf_map)
        
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Graph contains cycles after resolution")
        
        # 후처리
        self.dag_processor.post_process_graph(dag)
        dag = self.dag_processor.apply_rule_based_pruning(dag)
        
        print(f"   ✅ Final DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
        print(f"   📊 Node IDs: {list(dag.nodes())}")
        print(f"   📊 Edge pairs: {list(dag.edges())}")
        
        metrics = {
            "llm_dependency_calls": n_pairs,
            "llm_resource_calls": 0,  # 리소스 충돌은 간소화
            "total_llm_calls": n_pairs,
            "analysis_time": analysis_time,
            "pairs_per_second": n_pairs / analysis_time if analysis_time > 0 else 0,
            "batch_size": self.BATCH_SIZE,
            "parallel_optimization": True
        }
        
        return dag, metrics