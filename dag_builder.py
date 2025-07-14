# dag_builder.py
import json
import itertools
from typing import List, Dict
import networkx as nx

from models import SubTask
from utils import gpt


# ------------- Stage 2: DAG & parallel/serial analysis ---------- #

EDGE_TEMPLATE = """
For the following two sub‑tasks from the same parent task, answer with STRICT JSON "yes" or "no" only:

Sub‑task A: "{a}"
Sub‑task B: "{b}"

Question: Can B start before A is completed?
"""

RESOURCE_DEPENDENCY_TEMPLATE = """
Analyze if these two sub-tasks have any shared resource dependencies or conflicts.
Return JSON with "resource_conflict": true/false and "shared_resources": [list of shared resources].

Sub‑task A: "{a}"
Sub‑task B: "{b}"
"""

CONFIDENCE_TEMPLATE = """
Rate your confidence (0.0 to 1.0) in the dependency relationship between these tasks:

Sub‑task A: "{a}"  
Sub‑task B: "{b}"

Return only a number between 0.0 and 1.0.
"""


def build_dag(subtasks: List[SubTask]) -> nx.DiGraph:
    G = nx.DiGraph()
    for st in subtasks:
        G.add_node(st.id, obj=st)
    
    # Build dependency matrix with confidence scores
    dependency_matrix = {}
    for a, b in itertools.permutations(subtasks, 2):
        # Check basic dependency
        ans = gpt(EDGE_TEMPLATE.format(a=a.description, b=b.description),
                  system="Answer with exactly yes or no.", temperature=0)
        
        if ans.lower().startswith("no"):
            # Get confidence score for this dependency
            try:
                confidence = float(gpt(CONFIDENCE_TEMPLATE.format(a=a.description, b=b.description),
                                     system="Return only a number.", temperature=0))
            except:
                confidence = 0.7  # default confidence
            
            # Check resource conflicts
            try:
                resource_info = json.loads(gpt(RESOURCE_DEPENDENCY_TEMPLATE.format(a=a.description, b=b.description),
                                             system="Return valid JSON only.", temperature=0))
                has_conflict = resource_info.get("resource_conflict", False)
                shared_resources = resource_info.get("shared_resources", [])
            except:
                has_conflict = False
                shared_resources = []
            
            # Add edge with metadata
            if confidence > 0.5:  # Only add high-confidence dependencies
                G.add_edge(a.id, b.id, 
                          confidence=confidence,
                          has_resource_conflict=has_conflict,
                          shared_resources=shared_resources)
                dependency_matrix[(a.id, b.id)] = confidence
    
    # Cycle detection and resolution
    G = resolve_cycles_intelligently(G, dependency_matrix)
    
    # Add parallel block identification
    G = identify_parallel_blocks(G)
    
    return G


def resolve_cycles_intelligently(G: nx.DiGraph, dependency_matrix: Dict) -> nx.DiGraph:
    try:
        while True:
            cycles = list(nx.find_cycle(G, orientation="original"))
            if not cycles:
                break
            
            # Find the weakest edge in the cycle (lowest confidence)
            weakest_edge = None
            min_confidence = float('inf')
            
            for edge in cycles:
                edge_key = (edge[0], edge[1])
                confidence = dependency_matrix.get(edge_key, 0.5)
                if confidence < min_confidence:
                    min_confidence = confidence
                    weakest_edge = edge_key
            
            if weakest_edge:
                G.remove_edge(weakest_edge[0], weakest_edge[1])
                print(f"Removed cycle edge {weakest_edge} with confidence {min_confidence}")
            
    except nx.NetworkXNoCycle:
        pass
    
    return G


def identify_parallel_blocks(G: nx.DiGraph) -> nx.DiGraph:
    # Find levels of parallelism
    levels = {}
    for node in nx.topological_sort(G):
        predecessors = list(G.predecessors(node))
        if not predecessors:
            levels[node] = 0
        else:
            levels[node] = max(levels[pred] for pred in predecessors) + 1
    
    # Group nodes by level (potential parallel blocks)
    parallel_blocks = {}
    for node, level in levels.items():
        if level not in parallel_blocks:
            parallel_blocks[level] = []
        parallel_blocks[level].append(node)
    
    # Annotate nodes with parallel block info
    for level, nodes in parallel_blocks.items():
        for node in nodes:
            G.nodes[node]['parallel_level'] = level
            G.nodes[node]['parallel_peers'] = [n for n in nodes if n != node]
    
    # Calculate critical path
    critical_path = nx.algorithms.dag.longest_path(G)
    for i, node in enumerate(critical_path):
        G.nodes[node]['on_critical_path'] = True
        G.nodes[node]['critical_path_position'] = i
    
    return G
