# approaches/hierarchical.py
from typing import Dict, Any, List, Tuple
import networkx as nx

from models import SubTask
from .base import DAGApproach
from prompts import PromptManager


class HierarchicalApproach(DAGApproach):
    def __init__(self):
        super().__init__("hierarchical")
    
    def _build_dag_impl(self, subtasks: List[SubTask], original_task: str) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """Build DAG using hierarchical decomposition."""
        
        print("🌳 HIERARCHICAL APPROACH - Starting hierarchical decomposition...")
        print(f"   Original task: {original_task[:100]}...")
        
        cross_tree_calls = 0
        
        # 1. Try new hierarchical decomposition approach first
        try:
            print("   📤 Sending LLM request for hierarchical decomposition...")
            tree_data = self._decompose_hierarchical(original_task)
            print(f"   📥 LLM Response received. Tree structure:")
            self._print_tree(tree_data, indent="      ")
            
            print("   🔧 Converting tree to subtasks and DAG...")
            subtasks_from_tree = self._tree_to_subtasks(tree_data)
            dag, cross_tree_calls = self._tree_to_dag_new(tree_data, subtasks_from_tree)
            
            print(f"   ✅ DAG created: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
            print(f"   📊 DAG nodes: {list(dag.nodes())}")
            
            total_llm_calls = 1 + cross_tree_calls  # Decomposition + cross-tree analysis
            fallback_used = False
            approach_method = "hierarchical_decompose"
            
        except Exception as e:
            print(f"   ❌ New hierarchical decomposition failed: {e}")
            print("   🔄 Trying legacy organization approach...")
            
            # Fallback to legacy approach
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
                approach_method = "hierarchical_organize"
                
            except Exception as e2:
                print(f"   ❌ Legacy hierarchical approach also failed: {e2}")
                print("   🔄 Using sequential fallback...")
                # Final fallback: create simple sequential structure
                dag = self._create_sequential_dag(subtasks)
                total_llm_calls = 0
                fallback_used = True
                approach_method = "sequential_fallback"
        
        metrics = {
            "llm_organization_calls": 1 if not fallback_used else 0,
            "llm_cross_tree_calls": cross_tree_calls if not fallback_used else 0,
            "total_llm_calls": total_llm_calls,
            "approach_specific": approach_method,
            "fallback_used": fallback_used
        }
        
        return dag, metrics
    
    def _print_tree(self, node: Dict[str, Any], indent: str = ""):
        """Print tree structure for debugging."""
        print(f"{indent}{node['id']}: {node['desc'][:50]}...")
        for child in node.get('children', []):
            self._print_tree(child, indent + "  ")
    
    def _decompose_hierarchical(self, task: str) -> Dict[str, Any]:
        """Decompose task into hierarchical tree structure using the new template."""
        prompt = PromptManager.format_hierarchical_decompose_prompt(task)
        return self.llm_client.call_json(prompt, validator=self._validate_tree_json)
    
    def _organize_hierarchical(self, task: str, subtasks: List[SubTask]) -> Dict[str, Any]:
        """Organize subtasks into hierarchical tree (legacy method)."""
        subtask_list = "\n".join([f"{st.id}: {st.description}" for st in subtasks])
        prompt = PromptManager.format_hierarchical_organization_prompt(task, subtask_list)
        return self.llm_client.call_json(prompt, validator=self._validate_tree_json)
    
    def _tree_to_subtasks(self, tree: Dict[str, Any]) -> List[SubTask]:
        """Convert tree structure to flat list of SubTasks."""
        subtasks = []
        
        def traverse(node):
            # Only add non-root nodes as subtasks
            if node["id"] != "ROOT":
                subtasks.append(SubTask(id=node["id"], description=node["desc"]))
            
            for child in node["children"]:
                traverse(child)
        
        traverse(tree)
        return subtasks
    
    def _tree_to_dag_new(self, tree: Dict[str, Any], subtasks: List[SubTask]) -> Tuple[nx.DiGraph, int]:
        """Convert hierarchical tree to DAG with parent->child dependencies (new method)."""
        dag = nx.DiGraph()
        
        # Create mapping from ID to SubTask for adding obj attributes
        subtask_map = {st.id: st for st in subtasks}
        
        def add_nodes_and_edges(node, parent_id=None):
            node_id = node["id"]
            if node_id != "ROOT":  # Skip root node
                # Add node with both description and obj attributes
                subtask_obj = subtask_map.get(node_id, SubTask(id=node_id, description=node["desc"]))
                dag.add_node(node_id, description=node["desc"], obj=subtask_obj)
                
                # Add edges from parent to this node (dependency relationship)
                if parent_id and node_id != "ROOT":
                    dag.add_edge(parent_id, node_id, confidence=1.0, tree_edge=True)
            
            # Recursively process children
            current_parent = node_id if node_id != "ROOT" else parent_id
            for child in node["children"]:
                add_nodes_and_edges(child, current_parent)
        
        print(f"      Processing tree structure...")
        add_nodes_and_edges(tree)
        
        # Add cross-tree dependencies using selective LLM calls
        cross_tree_calls = self._add_cross_tree_dependencies_new(dag, tree)
        
        # Mandatory cycle removal after cross-tree dependencies
        dag = self._resolve_hierarchical_cycles(dag)
        
        # Apply rule-based pruning for formatting tasks
        dag = self.dag_processor.apply_rule_based_pruning(dag)
        
        # Apply post-processing
        self.dag_processor.post_process_graph(dag)
        
        return dag, cross_tree_calls
    
    def _add_cross_tree_dependencies_new(self, dag: nx.DiGraph, tree: Dict[str, Any]) -> int:
        """Add dependencies between different branches of the tree (new method)."""
        import itertools
        
        nodes = list(dag.nodes())
        cross_tree_calls = 0
        
        # Only check dependencies between nodes that are not in parent-child relationship
        for a, b in itertools.combinations(nodes, 2):
            if not (dag.has_edge(a, b) or dag.has_edge(b, a)):
                # Check if they're in different subtrees
                if self._are_in_different_subtrees(a, b, tree):
                    # Use LLM to determine cross-tree dependencies
                    subtask_a = dag.nodes[a]['obj']
                    subtask_b = dag.nodes[b]['obj']
                    
                    dep_ab, conf_ab = self._ask_cross_tree_dependency(subtask_a, subtask_b)
                    cross_tree_calls += 1
                    
                    dep_ba, conf_ba = self._ask_cross_tree_dependency(subtask_b, subtask_a)
                    cross_tree_calls += 1
                    
                    # Add edge for higher confidence dependency
                    if dep_ab and conf_ab >= 0.5:
                        if not dep_ba or conf_ab >= conf_ba:
                            # Add resource conflict metadata for cross-tree edges
                            has_conflict, conflict_resources = self._ask_resource_conflict(subtask_a, subtask_b)
                            cross_tree_calls += 1
                            dag.add_edge(a, b, confidence=conf_ab, cross_tree=True,
                                       has_resource_conflict=has_conflict, shared_resources=conflict_resources)
                    elif dep_ba and conf_ba >= 0.5:
                        # Add resource conflict metadata for cross-tree edges
                        has_conflict, conflict_resources = self._ask_resource_conflict(subtask_b, subtask_a)
                        cross_tree_calls += 1
                        dag.add_edge(b, a, confidence=conf_ba, cross_tree=True,
                                   has_resource_conflict=has_conflict, shared_resources=conflict_resources)
        
        return cross_tree_calls
    
    def _are_in_different_subtrees(self, node_a: str, node_b: str, tree: Dict[str, Any]) -> bool:
        """Check if two nodes are in different subtrees."""
        def find_subtree_root(node_id, current_node):
            if current_node["id"] == node_id:
                return current_node["id"]
            for child in current_node["children"]:
                result = find_subtree_root(node_id, child)
                if result:
                    return current_node["id"] if current_node["id"] != "ROOT" else result
            return None
        
        root_a = find_subtree_root(node_a, tree)
        root_b = find_subtree_root(node_b, tree)
        return root_a != root_b
    
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
                    # This is a grouping node - create a placeholder and mark as group
                    dag.add_node(node_id, description=node["desc"], obj=SubTask(id=node_id, description=node["desc"], mode="DEEP"), is_group=True)
                
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
        """Identify which subtasks belong to which tree groups based on hierarchical structure."""
        import networkx as nx
        
        groups = {}
        
        # Find all actual subtasks (nodes starting with 'S')
        subtasks = [node for node in dag.nodes() if node.startswith('S')]
        
        # Find grouping nodes (nodes that are not subtasks and not ROOT)
        group_nodes = [node for node in dag.nodes() 
                      if not node.startswith('S') and node != 'ROOT']
        
        print(f"      Found subtasks: {subtasks}")
        print(f"      Found group nodes: {group_nodes}")
        
        # For each group node, find its subtask descendants
        for group_node in group_nodes:
            group_subtasks = []
            
            # Find all descendants of this group node that are subtasks
            try:
                descendants = nx.descendants(dag, group_node)
                for desc in descendants:
                    if desc.startswith('S'):
                        group_subtasks.append(desc)
                        
                if group_subtasks:
                    groups[group_node] = group_subtasks
                    print(f"      Group '{group_node}': {group_subtasks}")
                    
            except Exception as e:
                print(f"      Warning: Could not find descendants for {group_node}: {e}")
        
        # If no groups found with proper grouping nodes, try alternative approach
        if not groups:
            print(f"      No explicit group nodes found, analyzing tree structure...")
            
            # Find subtasks by their parent nodes (direct predecessors)
            for subtask in subtasks:
                try:
                    predecessors = list(dag.predecessors(subtask))
                    if predecessors:
                        parent = predecessors[0]  # Take first parent
                        if parent != 'ROOT' and not parent.startswith('S'):
                            # This parent is a group node
                            if parent not in groups:
                                groups[parent] = []
                            if subtask not in groups[parent]:
                                groups[parent].append(subtask)
                        
                except Exception as e:
                    print(f"      Warning: Could not analyze parent for {subtask}: {e}")
        
        # Final fallback: if still no proper groups, create artificial groups for cross-analysis
        if not groups:
            print(f"      No hierarchical groups found, creating artificial groups for analysis")
            all_subtasks = subtasks
            if len(all_subtasks) >= 2:
                mid = len(all_subtasks) // 2
                groups["group_a"] = all_subtasks[:mid]
                groups["group_b"] = all_subtasks[mid:]
        
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
