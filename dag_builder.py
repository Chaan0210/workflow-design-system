# dag_builder.py
import json
import itertools
from typing import List, Dict
from collections import defaultdict
import networkx as nx

from models import SubTask
from utils import gpt
from prompts import EDGE_TEMPLATE, RESOURCE_DEPENDENCY_TEMPLATE, CONFIDENCE_TEMPLATE


# ------------- Stage 2: DAG & parallel/serial analysis ---------- #

def parse_yes_no(response: str) -> bool:
    """
    Intelligently parses a string response to determine if it means "yes" or "no".
    Handles plain text, JSON, and markdown-formatted JSON.
    Returns True for "yes" (can start in parallel), False for "no" (is a dependency).
    """
    clean_response = response.strip().lower()

    # Handle markdown code blocks
    if clean_response.startswith("```json"):
        clean_response = clean_response[7:-4].strip()
    elif clean_response.startswith("```"):
        clean_response = clean_response[3:-3].strip()

    # Try parsing as JSON
    try:
        # Remove outer quotes if they exist (e.g., '"no"')
        if clean_response.startswith('"') and clean_response.endswith('"'):
            clean_response = clean_response[1:-1]

        data = json.loads(clean_response)
        if isinstance(data, dict):
            if data.get("no") is True: return False
            if data.get("yes") is False: return False
            if data.get("answer", "").lower() == "no": return False
            return True # Default to "yes"
        elif isinstance(data, str):
            return "no" not in data
    except (json.JSONDecodeError, TypeError):
        pass

    # Treat as plain text
    return not clean_response.startswith("no")


def build_dag(subtasks: List[SubTask]) -> nx.DiGraph:
    G = nx.DiGraph()
    for st in subtasks:
        G.add_node(st.id, obj=st)
    
    # Build dependency matrix with confidence scores
    dependency_matrix = {}
    print("\n--- BUILDING DAG ---")
    for a, b in itertools.permutations(subtasks, 2):
        print(f"\nAnalyzing dependency: {a.id} -> {b.id}")

        # Check basic dependency
        prompt = EDGE_TEMPLATE.format(a=a.description, b=b.description)
        print("\n--- DEPENDENCY PROMPT ---")
        print(prompt)
        print("-------------------------")
        ans_str = gpt(prompt,
                  system="Answer with exactly yes or no.", temperature=0)
        print("\n--- DEPENDENCY LLM RESPONSE ---")
        print(ans_str)
        print("-----------------------------")
        
        is_dependent = not parse_yes_no(ans_str)

        if is_dependent:
            print(f"Result: Tasks are dependent.")
            # Get confidence score for this dependency
            prompt = CONFIDENCE_TEMPLATE.format(a=a.description, b=b.description)
            print("\n--- CONFIDENCE PROMPT ---")
            print(prompt)
            print("-------------------------")
            try:
                confidence_str = gpt(prompt,
                                     system="Return only a number.", temperature=0)
                print("\n--- CONFIDENCE LLM RESPONSE ---")
                print(confidence_str)
                print("-----------------------------")
                confidence = float(confidence_str)
            except (ValueError, TypeError):
                print("Could not parse confidence, using default 0.7")
                confidence = 0.7  # default confidence
            
            # Check resource conflicts
            prompt = RESOURCE_DEPENDENCY_TEMPLATE.format(a=a.description, b=b.description)
            print("\n--- RESOURCE CONFLICT PROMPT ---")
            print(prompt)
            print("--------------------------------")
            try:
                resource_info_str = gpt(prompt,
                                             system="Return valid JSON only.", temperature=0)
                print("\n--- RESOURCE CONFLICT LLM RESPONSE ---")
                print(resource_info_str)
                print("------------------------------------")
                
                # Clean the response from markdown code blocks
                if resource_info_str.strip().startswith("```json"):
                    resource_info_str = resource_info_str.strip()[7:-4]

                resource_info = json.loads(resource_info_str)
                has_conflict = resource_info.get("resource_conflict", False)
                shared_resources = resource_info.get("shared_resources", [])
            except (json.JSONDecodeError, TypeError):
                print("Could not parse resource conflict JSON.")
                has_conflict = False
                shared_resources = []
            
            # Add edge with metadata
            if confidence > 0.5:  # Only add high-confidence dependencies
                print(f"\nAdding edge {a.id} -> {b.id} with confidence {confidence}")
                G.add_edge(a.id, b.id, 
                          confidence=confidence,
                          has_resource_conflict=has_conflict,
                          shared_resources=shared_resources)
                dependency_matrix[(a.id, b.id)] = confidence
            else:
                print(f"\nSkipping edge {a.id} -> {b.id} due to low confidence {confidence}")
        else:
            print(f"Result: Tasks are not dependent.")

    # Cycle detection and resolution
    G = resolve_cycles_intelligently(G, dependency_matrix)
    
    # Add parallel block identification
    G = identify_parallel_blocks(G)
    
    print("\n--- DAG BUILDING COMPLETE ---")
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
    levels = {}
    try:
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            if not preds:
                levels[node] = 0
            else:
                levels[node] = max(levels[p] for p in preds) + 1
    except nx.NetworkXUnfeasible as e:
        raise ValueError("Graph contains cycles. Run resolve_cycles_intelligently first.") from e

    parallel_blocks = defaultdict(list)
    for node, level in levels.items():
        parallel_blocks[level].append(node)

    for level, nodes in parallel_blocks.items():
        for node in nodes:
            G.nodes[node]['parallel_level'] = level
            G.nodes[node]['parallel_peers'] = [n for n in nodes if n != node]

    try:
        critical_path = nx.algorithms.dag.dag_longest_path(G)
    except (nx.NetworkXUnfeasible, nx.NetworkXError):
        critical_path = []

    for i, node in enumerate(critical_path):
        G.nodes[node]['on_critical_path'] = True
        G.nodes[node]['critical_path_position'] = i

    return G
