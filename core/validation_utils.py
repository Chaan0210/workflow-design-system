# core/validation_utils.py
from typing import Dict, List, Any, Tuple
import json
import networkx as nx

from models import SubTask
from core.llm_utils import LLMClient, PromptManager


class DAGValidator:
    """Unified DAG validation utilities."""
    
    @staticmethod
    def validate_dependency_json(d: dict) -> None:
        """Validate dependency JSON structure."""
        assert isinstance(d.get("dependent"), bool)
        assert isinstance(d.get("confidence"), (int, float))
        c = float(d["confidence"])
        if not (0.0 <= c <= 1.0):
            raise ValueError("confidence must be within [0,1]")
    
    @staticmethod
    def validate_resource_json(d: dict) -> None:
        """Validate resource conflict JSON structure."""
        assert isinstance(d.get("resource_conflict"), bool)
        assert isinstance(d.get("shared_resources"), list)
    
    @staticmethod
    def validate_matrix_json(d: dict) -> None:
        """Validate matrix JSON structure."""
        assert isinstance(d.get("matrix"), list)
        assert isinstance(d.get("confidence_matrix"), list)
        assert isinstance(d.get("task_order"), list)
        
        matrix = d["matrix"]
        conf_matrix = d["confidence_matrix"]
        n = len(matrix)
        
        assert len(conf_matrix) == n
        assert len(d["task_order"]) == n
        
        for i, row in enumerate(matrix):
            assert len(row) == n, f"Matrix row {i} has wrong length"
            assert len(conf_matrix[i]) == n, f"Confidence row {i} has wrong length"
            assert matrix[i][i] == 0, f"Diagonal element [{i}][{i}] should be 0"
    
    @staticmethod
    def validate_tree_json(d: dict) -> None:
        """Validate tree JSON structure."""
        assert isinstance(d.get("id"), str)
        assert isinstance(d.get("desc"), str)
        assert isinstance(d.get("children"), list)
        for child in d["children"]:
            DAGValidator.validate_tree_json(child)


class WorkflowValidator:
    """Unified workflow validation utilities."""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()
    
    def validate_workflow_structure(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall workflow structure."""
        issues = {
            "structural_errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        subtasks = workflow_data.get('subtasks', [])
        dag = workflow_data.get('dag')
        
        # Check basic structure
        if not subtasks:
            issues["structural_errors"].append("No sub-tasks generated")
        
        if dag is None:
            issues["structural_errors"].append("No DAG provided")
            return issues
        
        # Check DAG validity
        if not nx.is_directed_acyclic_graph(dag):
            issues["structural_errors"].append("Workflow contains cycles")
        
        # Check node-task consistency
        if dag.number_of_nodes() != len(subtasks):
            issues["warnings"].append("DAG nodes don't match sub-tasks count")
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(dag))
        if isolated_nodes:
            issues["warnings"].append(f"Isolated tasks found: {isolated_nodes}")
        
        # Check connectivity
        if dag.number_of_edges() == 0 and len(subtasks) > 1:
            issues["warnings"].append("No dependencies between tasks - workflow may be too fragmented")
        
        # Check for over-connectivity
        max_possible_edges = len(subtasks) * (len(subtasks) - 1) / 2
        if dag.number_of_edges() > max_possible_edges * 0.5:
            issues["suggestions"].append("High connectivity - consider simplifying dependencies")
        
        return issues
    
    def validate_quality_with_llm(self, main_task: str, subtasks: List[SubTask], 
                                 dependencies: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Validate workflow quality using LLM."""
        
        subtasks_info = [{"id": st.id, "desc": st.description, "mode": getattr(st, 'mode', None)} 
                        for st in subtasks]
        
        prompt = PromptManager.format_quality_validation_prompt(
            main_task, 
            json.dumps(subtasks_info, ensure_ascii=False),
            str(dependencies)
        )
        
        try:
            quality_analysis = self.llm_client.call_json(prompt, validator=self._validate_quality_json)
            
            return {
                "completeness_score": quality_analysis.get("completeness_score", 0.7),
                "coherence_score": quality_analysis.get("coherence_score", 0.7),
                "efficiency_score": quality_analysis.get("efficiency_score", 0.7),
                "feasibility_score": quality_analysis.get("feasibility_score", 0.7),
                "validation_errors": quality_analysis.get("validation_errors", []),
                "warnings": quality_analysis.get("warnings", []),
                "suggestions": quality_analysis.get("suggestions", []),
                "llm_validation_success": True
            }
        
        except Exception as e:
            # Fallback validation
            return {
                "completeness_score": min(1.0, len(subtasks) / 5.0),
                "coherence_score": 0.7,
                "efficiency_score": 0.7,
                "feasibility_score": 0.8,
                "validation_errors": [f"LLM validation failed: {str(e)}"],
                "warnings": [],
                "suggestions": ["Consider manual review due to validation failure"],
                "llm_validation_success": False
            }
    
    def validate_execution_feasibility(self, subtasks: List[SubTask], dag: nx.DiGraph) -> Dict[str, Any]:
        """Validate execution feasibility of the workflow."""
        issues = {
            "feasibility_errors": [],
            "performance_warnings": [],
            "optimization_suggestions": []
        }
        
        # Check for extremely long workflows
        if len(subtasks) > 15:
            issues["performance_warnings"].append("Large number of subtasks may impact performance")
        
        # Check critical path length
        if nx.is_directed_acyclic_graph(dag):
            try:
                critical_path = nx.dag_longest_path(dag)
                if len(critical_path) > len(subtasks) * 0.8:
                    issues["performance_warnings"].append("Workflow is highly sequential - limited parallelization")
                elif len(critical_path) < len(subtasks) * 0.3:
                    issues["optimization_suggestions"].append("Good parallelization potential detected")
            except:
                pass
        
        # Check for task mode distribution
        wide_tasks = [st for st in subtasks if getattr(st, 'mode', None) == 'WIDE']
        deep_tasks = [st for st in subtasks if getattr(st, 'mode', None) == 'DEEP']
        
        if len(wide_tasks) > len(subtasks) * 0.8:
            issues["performance_warnings"].append("High proportion of WIDE tasks - may require significant resources")
        elif len(deep_tasks) > len(subtasks) * 0.8:
            issues["optimization_suggestions"].append("Deep reasoning focus - consider sequential execution")
        
        return issues
    
    @staticmethod
    def _validate_quality_json(d: dict) -> None:
        """Validate quality analysis JSON structure."""
        required_fields = ["completeness_score", "coherence_score", "efficiency_score", "feasibility_score"]
        for field in required_fields:
            assert field in d, f"Missing required field: {field}"
            assert isinstance(d[field], (int, float)), f"{field} must be numeric"
            assert 0.0 <= d[field] <= 1.0, f"{field} must be between 0.0 and 1.0"
        
        # Optional list fields
        list_fields = ["validation_errors", "warnings", "suggestions"]
        for field in list_fields:
            if field in d:
                assert isinstance(d[field], list), f"{field} must be a list"


class QualityMetricsValidator:
    """Validator for quality metrics and thresholds."""
    
    # Quality thresholds
    QUALITY_THRESHOLDS = {
        "completeness_score": {"poor": 0.3, "good": 0.7, "excellent": 0.9},
        "coherence_score": {"poor": 0.4, "good": 0.7, "excellent": 0.9},
        "efficiency_score": {"poor": 0.3, "good": 0.6, "excellent": 0.8},
        "feasibility_score": {"poor": 0.5, "good": 0.8, "excellent": 0.95},
        "overall_quality": {"poor": 0.4, "good": 0.7, "excellent": 0.85}
    }
    
    @classmethod
    def assess_quality_level(cls, metric_name: str, value: float) -> str:
        """Assess quality level for a given metric."""
        thresholds = cls.QUALITY_THRESHOLDS.get(metric_name, {"poor": 0.3, "good": 0.7, "excellent": 0.9})
        
        if value >= thresholds["excellent"]:
            return "excellent"
        elif value >= thresholds["good"]:
            return "good"
        elif value >= thresholds["poor"]:
            return "fair"
        else:
            return "poor"
    
    @classmethod
    def generate_quality_report(cls, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive quality assessment report."""
        report = {
            "overall_assessment": cls.assess_quality_level("overall_quality", 
                                                         quality_metrics.get("overall_quality", 0.0)),
            "metric_assessments": {},
            "recommendations": [],
            "critical_issues": []
        }
        
        for metric, value in quality_metrics.items():
            if metric in cls.QUALITY_THRESHOLDS:
                assessment = cls.assess_quality_level(metric, value)
                report["metric_assessments"][metric] = {
                    "value": value,
                    "level": assessment
                }
                
                # Generate recommendations based on poor metrics
                if assessment == "poor":
                    if metric == "completeness_score":
                        report["recommendations"].append("Consider adding more comprehensive subtasks")
                    elif metric == "coherence_score":
                        report["recommendations"].append("Review and improve dependency relationships")
                    elif metric == "efficiency_score":
                        report["recommendations"].append("Optimize workflow for better parallelization")
                    elif metric == "feasibility_score":
                        report["critical_issues"].append("Workflow feasibility concerns detected")
        
        # Overall recommendations
        if report["overall_assessment"] == "poor":
            report["critical_issues"].append("Overall workflow quality is poor - consider redesigning")
        elif report["overall_assessment"] == "fair":
            report["recommendations"].append("Workflow quality is adequate but has room for improvement")
        
        return report
