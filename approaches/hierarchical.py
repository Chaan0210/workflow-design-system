# approaches/hierarchical.py
from typing import Dict, Any, List, Tuple
import networkx as nx

from models import SubTask
from .base import DAGApproach
from prompts import PromptManager


class HierarchicalApproach(DAGApproach):
    """
    Hierarchical DAG building approach.
    
    Uses hierarchical tree decomposition with cross-tree dependency analysis:
    1. Organize existing subtasks into hierarchical tree structure
    2. Add LLM-based cross-tree dependency analysis
    3. Apply resource conflict checking
    4. Resolve cycles by removing lowest confidence cross-tree edges
    """
    
    def __init__(self):
        super().__init__("hierarchical")
    
    def _build_dag_impl(self, subtasks: List[SubTask], original_task: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """Build DAG using hierarchical decomposition."""
        
        print("🌳 HIERARCHICAL APPROACH - Starting hierarchical decomposition...")
        print(f"   Original task: {original_task[:100]}...")
        
        cross_tree_calls = 0  # Initialize here
        
        # 1. Try to decompose hierarchically
        try:
            print("   📤 Sending LLM request for hierarchical organization...")
            tree_data = self._organize_hierarchical(original_task, subtasks)
            print(f"   📥 LLM Response received. Tree structure:")
            self._print_tree(tree_data, indent="      ")
            
            print("   🔧 Converting tree to DAG...")
            dag, cross_tree_calls = self._tree_to_dag(tree_data, subtasks)
            print(f"   ✅ DAG created: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
            print(f"   📊 DAG nodes: {list(dag.nodes())}")
            
            total_llm_calls = 1 + cross_tree_calls  # Organization + cross-tree analysis
            fallback_used = False
            
        except Exception as e:
            print(f"   ❌ Hierarchical decomposition failed: {e}")
            print("   🔄 Using sequential fallback...")
            # Fallback: create simple sequential structure
            dag = self._create_sequential_dag(subtasks)
            total_llm_calls = 0
            fallback_used = True
        
        metrics = {
            "llm_organization_calls": 1 if not fallback_used else 0,
            "llm_cross_tree_calls": cross_tree_calls if not fallback_used else 0,
            "total_llm_calls": total_llm_calls,
            "approach_specific": "hierarchical_tree",
            "fallback_used": fallback_used
        }
        
        return dag, metrics
    
    def _print_tree(self, node: Dict[str, Any], indent: str = ""):
        """Print tree structure for debugging."""
        print(f"{indent}{node['id']}: {node['desc'][:50]}...")
        for child in node.get('children', []):
            self._print_tree(child, indent + "  ")
    
    def _organize_hierarchical(self, task: str, subtasks: List[SubTask]) -> Dict[str, Any]:
        """Organize subtasks into hierarchical tree."""
        subtask_list = "\n".join([f"{st.id}: {st.description}" for st in subtasks])
        prompt = PromptManager.format_hierarchical_organization_prompt(task, subtask_list)
        return self.llm_client.call_json(prompt, validator=self._validate_tree_json)
    
    def _tree_to_dag(self, tree: Dict[str, Any], subtasks: List[SubTask]) -> Tuple[nx.DiGraph, int]:
        """Convert tree to DAG with hierarchical parent->child edges."""
        dag = nx.DiGraph()
        subtask_map = {st.id: st for st in subtasks}
        actual_subtasks = []  # Track which nodes are actual subtasks (not groups)
        
        def add_nodes_and_edges(node, parent_id=None):
            node_id = node["id"]
            if node_id != "ROOT":
                # Check if this is an actual subtask or just a grouping node
                if node_id in subtask_map:
                    # This is an actual subtask
                    subtask_obj = subtask_map[node_id]
                    dag.add_node(node_id, description=subtask_obj.description, obj=subtask_obj)
                    actual_subtasks.append(node_id)
                else:
                    # This is a grouping node - create a placeholder
                    dag.add_node(node_id, description=node["desc"], obj=SubTask(id=node_id, description=node["desc"], mode="DEEP"))
                
                # Add parent->child edge (hierarchical dependency)
                if parent_id and parent_id != "ROOT":
                    dag.add_edge(parent_id, node_id, confidence=0.9, tree_edge=True)
            
            # Process children
            current_parent = node_id if node_id != "ROOT" else parent_id
            for child in node["children"]:
                add_nodes_and_edges(child, current_parent)
        
        print(f"      Processing tree structure...")
        add_nodes_and_edges(tree)
        
        # Add selective cross-dependencies between different sub-trees
        print(f"      Adding cross-tree dependencies for actual subtasks...")
        cross_tree_calls = self._add_cross_tree_dependencies(dag, actual_subtasks, subtasks)
        
        # Remove cycles before post-processing
        dag = self._resolve_hierarchical_cycles(dag)
        
        # Apply post-processing
        self.dag_processor.post_process_graph(dag)
        dag = self.dag_processor.apply_rule_based_pruning(dag)
        
        return dag, cross_tree_calls
    
    def _add_cross_tree_dependencies(self, dag: nx.DiGraph, actual_subtasks: List[str], all_subtasks: List[SubTask]) -> int:
        """Add selective cross-tree dependencies between subtasks from different groups."""
        subtask_map = {st.id: st for st in all_subtasks}
        cross_tree_calls = 0
        
        # Find subtasks from different sub-trees
        tree_groups = self._get_tree_groups(dag)
        print(f"      Tree groups: {tree_groups}")
        
        # Only check dependencies between different sub-trees
        for group1, tasks1 in tree_groups.items():
            for group2, tasks2 in tree_groups.items():
                if group1 != group2:
                    for task1 in tasks1:
                        for task2 in tasks2:
                            if task1 in subtask_map and task2 in subtask_map:
                                # Ask LLM for cross-tree dependency
                                has_dep, confidence = self._ask_cross_tree_dependency(
                                    subtask_map[task1], subtask_map[task2]
                                )
                                cross_tree_calls += 1
                                
                                if has_dep and confidence >= 0.5:
                                    print(f"         Cross-tree dependency: {task1} → {task2} (conf: {confidence})")
                                    
                                    # Also check for resource conflicts
                                    has_conflict, shared_resources = self._ask_resource_conflict(
                                        subtask_map[task1], subtask_map[task2]
                                    )
                                    cross_tree_calls += 1
                                    
                                    dag.add_edge(task1, task2, 
                                               confidence=confidence,
                                               cross_tree_edge=True,
                                               has_resource_conflict=has_conflict,
                                               shared_resources=shared_resources)
        
        print(f"      Cross-tree LLM calls: {cross_tree_calls}")
        return cross_tree_calls
    
    def _get_tree_groups(self, dag: nx.DiGraph) -> Dict[str, List[str]]:
        """Identify which subtasks belong to which tree groups."""
        groups = {}
        
        for node in dag.nodes():
            # Find all predecessors (parents) of this node
            predecessors = list(dag.predecessors(node))
            
            # If node has no predecessors or its predecessor is a grouping node, it's a group leader
            if not predecessors:
                continue
                
            parent = predecessors[0] if predecessors else None
            if parent and not parent.startswith('S'):  # Grouping nodes don't start with 'S'
                if parent not in groups:
                    groups[parent] = []
                groups[parent].append(node)
            
        return groups
    
    def _ask_cross_tree_dependency(self, task_a: SubTask, task_b: SubTask) -> Tuple[bool, float]:
        """Ask LLM about cross-tree dependency between two subtasks."""
        prompt = PromptManager.format_cross_tree_dependency_prompt(task_a.description, task_b.description)
        try:
            result = self.llm_client.call_json(prompt)
            return bool(result.get("dependent", False)), float(result.get("confidence", 0.5))
        except Exception:
            return False, 0.5
    
    def _ask_resource_conflict(self, task_a: SubTask, task_b: SubTask) -> Tuple[bool, List[str]]:
        """Ask LLM about resource conflicts between two subtasks."""
        prompt = PromptManager.format_resource_conflict_analysis_prompt(task_a.description, task_b.description)
        try:
            result = self.llm_client.call_json(prompt)
            return bool(result.get("has_conflict", False)), result.get("shared_resources", [])
        except Exception:
            return False, []
    
    def _resolve_hierarchical_cycles(self, dag: nx.DiGraph) -> nx.DiGraph:
        """Remove cycles from hierarchical DAG by removing lowest confidence cross-tree edges."""
        print(f"      Checking for cycles...")
        
        while not nx.is_directed_acyclic_graph(dag):
            try:
                cycle = nx.find_cycle(dag, orientation="original")
                print(f"         Found cycle: {[edge[:2] for edge in cycle]}")
                
                # Find the weakest cross-tree edge in the cycle
                weakest_edge = None
                min_confidence = float('inf')
                
                for u, v, _ in cycle:
                    edge_data = dag[u][v]
                    if edge_data.get('cross_tree_edge', False):
                        confidence = edge_data.get('confidence', 0.5)
                        if confidence < min_confidence:
                            min_confidence = confidence
                            weakest_edge = (u, v)
                
                # If no cross-tree edge found, remove any edge with lowest confidence
                if not weakest_edge:
                    for u, v, _ in cycle:
                        edge_data = dag[u][v]
                        confidence = edge_data.get('confidence', 0.5)
                        if confidence < min_confidence:
                            min_confidence = confidence
                            weakest_edge = (u, v)
                
                if weakest_edge:
                    print(f"         Removing weakest edge: {weakest_edge} (conf: {min_confidence})")
                    dag.remove_edge(*weakest_edge)
                else:
                    # Fallback: remove first edge in cycle
                    first_edge = cycle[0][:2]
                    print(f"         Removing first edge in cycle: {first_edge}")
                    dag.remove_edge(*first_edge)
                    
            except nx.NetworkXNoCycle:
                break
        
        print(f"      ✅ DAG is now acyclic")
        return dag
    
    def _validate_tree_json(self, d: dict) -> None:
        """Validate tree JSON structure."""
        if not isinstance(d, dict):
            raise ValueError(f"Expected dict, got {type(d)}")
        
        if not isinstance(d.get("id"), str):
            raise ValueError(f"Missing or invalid 'id' field: {d.get('id')}")
            
        if not isinstance(d.get("desc"), str):
            raise ValueError(f"Missing or invalid 'desc' field: {d.get('desc')}")
            
        if not isinstance(d.get("children"), list):
            raise ValueError(f"Missing or invalid 'children' field: {d.get('children')}")
        
        # Recursively validate children
        for i, child in enumerate(d["children"]):
            try:
                self._validate_tree_json(child)
            except Exception as e:
                raise ValueError(f"Child {i} validation failed: {e}")
