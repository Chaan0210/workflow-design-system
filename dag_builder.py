# dag_builder.py
import json
import itertools
from typing import List, Dict, Tuple
from collections import defaultdict
import networkx as nx

from models import SubTask
from utils import call_gpt_json
from prompts import DEPENDENCY_STRICT_TEMPLATE, RESOURCE_DEPENDENCY_STRICT_TEMPLATE

LOW_CONF_L = 0.4
CUTOFF = 0.5

def _validate_dependency_json(d: dict):
    assert isinstance(d.get("dependent"), bool)
    assert isinstance(d.get("confidence"), (int, float))
    c = float(d["confidence"])
    if not (0.0 <= c <= 1.0):
        raise ValueError("confidence must be within [0,1]")

def _validate_resource_json(d: dict):
    assert isinstance(d.get("resource_conflict"), bool)
    assert isinstance(d.get("shared_resources"), list)

def ask_dependency(a: SubTask, b: SubTask) -> Tuple[bool, float, str]:
    prompt = DEPENDENCY_STRICT_TEMPLATE.format(a=a.description, b=b.description)
    data = call_gpt_json(
        prompt,
        system="Return ONLY valid JSON. No prose.",
        validator=_validate_dependency_json,
    )
    return bool(data["dependent"]), float(data["confidence"]), data.get("reason", "")

def ask_resource_conflict(a: SubTask, b: SubTask) -> Tuple[bool, List[str]]:
    prompt = RESOURCE_DEPENDENCY_STRICT_TEMPLATE.format(a=a.description, b=b.description)
    data = call_gpt_json(
        prompt,
        system="Return ONLY valid JSON. No prose.",
        validator=_validate_resource_json,
    )
    return bool(data["resource_conflict"]), list(data.get("shared_resources", []))

def resolve_cycles_intelligently(G: nx.DiGraph, edge_conf: Dict[Tuple[str, str], float]) -> nx.DiGraph:
    try:
        while True:
            cycles = list(nx.find_cycle(G, orientation="original"))
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
                G.remove_edge(*weakest)
    except nx.NetworkXNoCycle:
        pass
    return G

def post_process_graph(G: nx.DiGraph) -> None:
    # 1) confidence rounding/format
    for u, v, d in G.edges(data=True):
        if "confidence" in d:
            d["confidence"] = round(float(d["confidence"]), 3)

    # 2) ensure critical path, parallel levels
    try:
        critical_path = nx.algorithms.dag.dag_longest_path(G)
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        critical_path = []

    for n in G.nodes():
        G.nodes[n]['on_critical_path'] = False
        G.nodes[n]['critical_path_position'] = None

    for i, n in enumerate(critical_path):
        G.nodes[n]['on_critical_path'] = True
        G.nodes[n]['critical_path_position'] = i

    # 3) parallel levels
    try:
        levels = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1

        parallel_blocks = defaultdict(list)
        for node, level in levels.items():
            parallel_blocks[level].append(node)

        for level, nodes in parallel_blocks.items():
            for node in nodes:
                G.nodes[node]['parallel_level'] = level
                G.nodes[node]['parallel_peers'] = [n for n in nodes if n != node]
    except nx.NetworkXUnfeasible:
        # 이미 cycle 처리 실패 시 예외 던져서 상위에서 잡게 하는 편이 안전
        raise ValueError("Graph still contains cycles after post-process().")

def build_dag(subtasks: List[SubTask]) -> nx.DiGraph:
    G = nx.DiGraph()
    for st in subtasks:
        G.add_node(st.id, obj=st)

    edge_conf_map: Dict[Tuple[str, str], float] = {}
    low_conf_candidates: List[Tuple[str, str, float]] = []

    # --- 양방향 분석 ---
    for a, b in itertools.combinations(subtasks, 2):
        dep_ab, conf_ab, _ = ask_dependency(a, b)
        dep_ba, conf_ba, _ = ask_dependency(b, a)

        # 충돌 해소
        if dep_ab and dep_ba:
            # 둘 다 의존이라면 더 높은 confidence만 남기고 나머지 버림
            if conf_ab >= conf_ba:
                dep_ba = False
            else:
                dep_ab = False

        # AB edge 처리
        if dep_ab:
            has_conflict, shared = ask_resource_conflict(a, b)
            if conf_ab >= CUTOFF:
                G.add_edge(a.id, b.id,
                           confidence=conf_ab,
                           has_resource_conflict=has_conflict,
                           shared_resources=shared)
                edge_conf_map[(a.id, b.id)] = conf_ab
            elif conf_ab >= LOW_CONF_L:
                low_conf_candidates.append((a.id, b.id, conf_ab))

        # BA edge 처리
        if dep_ba:
            has_conflict, shared = ask_resource_conflict(b, a)
            if conf_ba >= CUTOFF:
                G.add_edge(b.id, a.id,
                           confidence=conf_ba,
                           has_resource_conflict=has_conflict,
                           shared_resources=shared)
                edge_conf_map[(b.id, a.id)] = conf_ba
            elif conf_ba >= LOW_CONF_L:
                low_conf_candidates.append((b.id, a.id, conf_ba))

    # --- 후처리 보정 패스: low_conf_candidates 중 cycle을 만들지 않는 한도에서 추가 ---
    # 규칙: 동일 노드 pair가 이미 다른 방향으로 연결돼 있으면 생략.
    for u, v, c in sorted(low_conf_candidates, key=lambda x: -x[2]):
        if not G.has_edge(u, v) and not G.has_edge(v, u):
            G.add_edge(u, v,
                       confidence=round(c, 3),
                       has_resource_conflict=False,
                       shared_resources=[])

    # --- 사이클 제거 ---
    G = resolve_cycles_intelligently(G, edge_conf_map)

    # --- 여전히 사이클이면 에러 ---
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph still contains cycles after resolution.")

    # --- 일관된 post process ---
    post_process_graph(G)

    return G
