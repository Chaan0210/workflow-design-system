# core/json_utils.py
import json
import datetime
from typing import Any, Dict, List, Set
import networkx as nx


class JsonSerializer:
    """Unified JSON serialization handler for all workflow objects."""
    
    @staticmethod
    def is_serializable(obj: Any) -> bool:
        """Check if object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def make_serializable(obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: JsonSerializer.make_serializable(v) for k, v in obj.items() 
                   if not (hasattr(v, 'nodes') and hasattr(v, 'edges'))}  # Skip DiGraph objects
        elif isinstance(obj, list):
            return [JsonSerializer.make_serializable(item) for item in obj]
        elif hasattr(obj, 'nodes') and hasattr(obj, 'edges'):  # DiGraph
            return JsonSerializer._digraph_to_dict(obj)
        elif hasattr(obj, 'id') and hasattr(obj, 'description'):  # SubTask
            return JsonSerializer._subtask_to_dict(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)
    
    @staticmethod
    def _digraph_to_dict(dag: nx.DiGraph) -> Dict[str, Any]:
        """Convert NetworkX DiGraph to serializable dictionary."""
        return {
            "nodes": [{
                "id": n, 
                **{k: v for k, v in data.items() 
                   if k != "obj" and isinstance(v, (str, int, float, bool, list, dict))}
            } for n, data in dag.nodes(data=True)],
            "edges": [{
                "source": u, 
                "target": v, 
                **JsonSerializer.make_serializable(data)
            } for u, v, data in dag.edges(data=True)]
        }
    
    @staticmethod
    def _subtask_to_dict(subtask) -> Dict[str, Any]:
        """Convert SubTask object to serializable dictionary."""
        return {
            "id": subtask.id,
            "description": subtask.description,
            "mode": getattr(subtask, 'mode', None)
        }
    
    @staticmethod
    def clean_for_json(obj: Any) -> Any:
        """Recursively clean object for JSON serialization.""" 
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if k == "dag" and hasattr(v, 'nodes'):  # Skip DiGraph objects
                    continue
                elif isinstance(v, dict):
                    cleaned[k] = JsonSerializer.clean_for_json(v)
                elif isinstance(v, list):
                    cleaned[k] = [JsonSerializer.clean_for_json(item) if isinstance(item, dict) else item 
                                 for item in v if JsonSerializer.is_serializable(item)]
                elif JsonSerializer.is_serializable(v):
                    cleaned[k] = v
            return cleaned
        return obj if JsonSerializer.is_serializable(obj) else None
    
    @staticmethod
    def save_json(data: Any, file_path: str, indent: int = 2):
        """Save data as JSON file with proper serialization."""
        serializable_data = JsonSerializer.make_serializable(data)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=indent, ensure_ascii=False)


# Convenience function for backward compatibility
def make_json_serializable(obj: Any) -> Any:
    """Make object JSON serializable (convenience function)."""
    return JsonSerializer.make_serializable(obj)
