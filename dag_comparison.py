# dag_comparison.py
import time
from typing import Dict, Any, List, Optional, Tuple

from models import SubTask
from workflow_engine import WorkflowEngine
from core import JsonSerializer, ComparisonEvaluator
from core.evaluation_utils import MetricsCalculator, QualityAssessor
from core.validation_utils import WorkflowValidator, QualityMetricsValidator

from approaches.bidirectional import BidirectionalApproach
from approaches.hierarchical import HierarchicalApproach  
from approaches.matrix import MatrixApproach


class DAGComparison:
    """Main DAG comparison framework."""
    
    def __init__(self):
        self.approaches = {
            "bidirectional": BidirectionalApproach(),
            "hierarchical": HierarchicalApproach(),
            "matrix": MatrixApproach()
        }
        self.evaluator = ComparisonEvaluator()
        self.workflow_validator = WorkflowValidator()
    
    def compare_approaches(self, task: str, subtasks: Optional[List[SubTask]] = None) -> Dict[str, Any]:
        """Compare all DAG building approaches."""
        
        # Use workflow engine for decomposition if needed
        if subtasks is None:
            print("📝 INITIAL DECOMPOSITION - Using WorkflowEngine...")
            print(f"   Original task: {task[:100]}...")
            engine = WorkflowEngine()
            subtasks = engine._decompose_task(task)
            print(f"   ✅ Decomposition complete: {len(subtasks)} subtasks created")
            print("   📋 Subtasks list:")
            for st in subtasks:
                print(f"      {st.id}: {st.description}")
                print(f"           Mode: {st.mode}")
        else:
            print(f"📝 Using provided subtasks ({len(subtasks)}):")
            for st in subtasks:
                print(f"      {st.id}: {st.description}")
                print(f"           Mode: {st.mode}")
        
        print(f"\n🔄 Comparing DAG approaches for {len(subtasks)} subtasks...")
        
        results = {}
        
        # Test each approach
        for name, approach in self.approaches.items():
            print(f"\n{'='*60}")
            print(f"🚀 Running {name.upper()} approach...")
            print('='*60)
            
            try:
                dag, metrics = approach.build_dag(subtasks, task)
                
                # Evaluate workflow quality and structure
                evaluation_results = self._evaluate_approach_workflow(
                    dag, subtasks, task, metrics, name
                )
                
                results[name] = {
                    "dag": dag,
                    "metrics": metrics,
                    "subtasks": subtasks,
                    "success": metrics.get("success", True),
                    "workflow_evaluation": evaluation_results
                }
                
                if metrics.get("success", True):
                    print(f"✅ {name.upper()} approach completed successfully")
                    print(f"   Build time: {metrics.get('build_time', 0):.3f}s")
                    print(f"   LLM calls: {metrics.get('total_llm_calls', 0)}")
                    if 'fallback_used' in metrics:
                        print(f"   Used fallback: {metrics['fallback_used']}")
                    
                    # Print workflow evaluation results
                    eval_results = evaluation_results
                    print(f"   📊 Workflow Evaluation:")
                    print(f"      Quality Score: {eval_results['quality_metrics']['overall_quality']:.3f}")
                    print(f"      Completeness: {eval_results['quality_metrics']['completeness_score']:.3f}")
                    print(f"      Coherence: {eval_results['quality_metrics']['coherence_score']:.3f}")
                    print(f"      Efficiency: {eval_results['quality_metrics']['efficiency_score']:.3f}")
                    print(f"      Feasibility: {eval_results['quality_metrics']['feasibility_score']:.3f}")
                    
                    # Print structural metrics
                    struct_metrics = eval_results['structural_metrics']
                    print(f"   🏗️  Structural Metrics:")
                    print(f"      Nodes: {struct_metrics['num_nodes']}, Edges: {struct_metrics['num_edges']}")
                    print(f"      Is DAG: {struct_metrics['is_dag']}")
                    print(f"      Density: {struct_metrics['density']:.3f}")
                    if struct_metrics.get('longest_path_len'):
                        print(f"      Longest Path: {struct_metrics['longest_path_len']} nodes")
                    
                    # Print complexity metrics
                    complexity = eval_results['complexity_metrics']
                    print(f"   🧠 Complexity Analysis:")
                    print(f"      Domain: {complexity['domain_complexity']:.1f}/10")
                    print(f"      Coordination: {complexity['coordination_complexity']:.1f}/10")
                    print(f"      Computational: {complexity['computational_complexity']:.1f}/10")
                    print(f"      Mode Heterogeneity: {complexity['mode_heterogeneity']:.3f}")
                    
                    # Print validation issues
                    validation = eval_results['validation_issues']
                    if validation['structural_errors']:
                        print(f"   ❌ Structural Errors: {len(validation['structural_errors'])}")
                    if validation['warnings']:
                        print(f"   ⚠️  Warnings: {len(validation['warnings'])}")
                    if validation['suggestions']:
                        print(f"   💡 Suggestions: {len(validation['suggestions'])}")
                else:
                    print(f"❌ {name.upper()} approach failed: {metrics.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ {name.upper()} approach failed: {str(e)}")
                results[name] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Generate comparison summary
        results["comparison_summary"] = self._generate_summary(results)
        
        # Generate comprehensive evaluation comparison
        results["evaluation_comparison"] = self._generate_evaluation_comparison(results)
        
        return results
    
    def evaluate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate comparison results."""
        return self.evaluator.evaluate_comparison_results(results)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file."""
        JsonSerializer.save_json(results, output_path)
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary."""
        import networkx as nx
        
        summary = {
            "successful_approaches": [],
            "performance_comparison": {},
            "structural_comparison": {}
        }
        
        for approach_name, result in results.items():
            if approach_name == "comparison_summary":
                continue
            
            if result.get("success", False):
                summary["successful_approaches"].append(approach_name)
                metrics = result["metrics"]
                
                summary["performance_comparison"][approach_name] = {
                    "build_time": metrics["build_time"],
                    "total_llm_calls": metrics.get("total_llm_calls", 0),
                    "nodes": metrics["nodes"],
                    "edges": metrics["edges"]
                }
                
                # Structural analysis
                dag = result["dag"]
                if nx.is_directed_acyclic_graph(dag) and dag.number_of_nodes() > 0:
                    try:
                        summary["structural_comparison"][approach_name] = {
                            "is_dag": True,
                            "longest_path": len(nx.dag_longest_path(dag)),
                            "average_degree": sum(dict(dag.degree()).values()) / dag.number_of_nodes(),
                            "density": nx.density(dag)
                        }
                    except:
                        summary["structural_comparison"][approach_name] = {"is_dag": True, "analysis_failed": True}
                else:
                    summary["structural_comparison"][approach_name] = {"is_dag": False}
        
        return summary
    
    def _evaluate_approach_workflow(self, dag, subtasks: List[SubTask], task: str, 
                                   build_metrics: Dict[str, Any], approach_name: str) -> Dict[str, Any]:
        """Comprehensive workflow evaluation for a single approach."""
        print(f"      🔍 Evaluating {approach_name} workflow...")
        
        # 1. Structural metrics using MetricsCalculator
        structural_metrics = MetricsCalculator.structural_metrics(dag)
        
        # 2. Quality assessment using QualityAssessor
        quality_metrics = QualityAssessor.assess_workflow_quality(
            {"subtasks": subtasks, "dag": dag}, subtasks, dag
        )
        
        # 3. Complexity metrics calculation
        complexity_metrics = MetricsCalculator.complexity_metrics({
            "subtasks": subtasks,
            "dag": dag
        })
        
        # 4. Workflow validation using WorkflowValidator
        validation_issues = self.workflow_validator.validate_workflow_structure({
            "subtasks": subtasks,
            "dag": dag
        })
        
        # 5. Execution feasibility validation
        execution_issues = self.workflow_validator.validate_execution_feasibility(subtasks, dag)
        
        # 6. Quality metrics validation and assessment
        quality_report = QualityMetricsValidator.generate_quality_report(quality_metrics)
        
        # 7. Execution metrics (if we have durations data)
        durations = {st.id: 1.0 for st in subtasks}  # Default duration
        execution_metrics = MetricsCalculator.execution_metrics(dag, durations)
        
        return {
            "structural_metrics": structural_metrics,
            "quality_metrics": quality_metrics,
            "complexity_metrics": complexity_metrics,
            "execution_metrics": execution_metrics,
            "validation_issues": validation_issues,
            "execution_feasibility": execution_issues,
            "quality_report": quality_report,
            "approach_name": approach_name,
            "build_metrics": build_metrics
        }
    
    def _generate_evaluation_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation comparison across approaches."""
        comparison = {
            "quality_comparison": {},
            "complexity_comparison": {},
            "structural_comparison": {},
            "execution_comparison": {},
            "recommendations": {}
        }
        
        successful_results = {}
        for approach_name, result in results.items():
            if approach_name in ["comparison_summary", "evaluation_comparison"]:
                continue
            if result.get("success", False) and "workflow_evaluation" in result:
                successful_results[approach_name] = result["workflow_evaluation"]
        
        if not successful_results:
            return comparison
        
        # Quality comparison
        print("\n📊 COMPREHENSIVE EVALUATION COMPARISON")
        print("=" * 80)
        
        for approach_name, eval_data in successful_results.items():
            quality = eval_data["quality_metrics"]
            complexity = eval_data["complexity_metrics"]
            structural = eval_data["structural_metrics"]
            execution = eval_data["execution_metrics"]
            
            comparison["quality_comparison"][approach_name] = quality
            comparison["complexity_comparison"][approach_name] = complexity
            comparison["structural_comparison"][approach_name] = structural
            comparison["execution_comparison"][approach_name] = execution
        
        # Generate recommendations using QualityAssessor
        quality_metrics_for_rec = {}
        for approach_name, eval_data in successful_results.items():
            metrics = {}
            # Flatten relevant metrics for recommendation
            metrics.update(eval_data["structural_metrics"])
            metrics.update(eval_data["execution_metrics"])
            quality_metrics_for_rec[approach_name] = metrics
        
        if quality_metrics_for_rec:
            recommendations = QualityAssessor.generate_recommendations(quality_metrics_for_rec)
            comparison["recommendations"] = recommendations
            
            # Print recommendations summary
            print("\n🎯 APPROACH RECOMMENDATIONS:")
            for use_case, rec in recommendations.items():
                print(f"\n{use_case.upper().replace('_', ' ')}:")
                print(f"   Recommended: {rec['recommended_approach']}")
                print(f"   Score: {rec['score']:.3f}")
                reasoning_lines = rec['reasoning'].split('\n')
                for line in reasoning_lines:
                    if line.strip():
                        print(f"   {line}")
        
        # Print detailed comparison table
        self._print_evaluation_table(successful_results)
        
        return comparison
    
    def _print_evaluation_table(self, successful_results: Dict[str, Any]):
        """Print a detailed evaluation comparison table."""
        print("\n📋 DETAILED EVALUATION TABLE:")
        print("-" * 100)
        
        # Header
        approaches = list(successful_results.keys())
        print(f"{'Metric':<30}", end="")
        for approach in approaches:
            print(f"{approach.upper():<15}", end="")
        print()
        print("-" * 100)
        
        # Quality metrics
        print("QUALITY METRICS:")
        quality_metrics = ["completeness_score", "coherence_score", "efficiency_score", "feasibility_score", "overall_quality"]
        for metric in quality_metrics:
            print(f"  {metric.replace('_', ' ').title():<28}", end="")
            for approach in approaches:
                value = successful_results[approach]["quality_metrics"].get(metric, 0)
                print(f"{value:<15.3f}", end="")
            print()
        
        print("\nSTRUCTURAL METRICS:")
        structural_metrics = ["num_nodes", "num_edges", "density", "average_degree"]
        for metric in structural_metrics:
            print(f"  {metric.replace('_', ' ').title():<28}", end="")
            for approach in approaches:
                value = successful_results[approach]["structural_metrics"].get(metric, 0)
                if isinstance(value, (int, float)):
                    print(f"{value:<15.3f}", end="")
                else:
                    print(f"{str(value):<15}", end="")
            print()
        
        print("\nCOMPLEXITY METRICS:")
        complexity_metrics = ["domain_complexity", "coordination_complexity", "computational_complexity", "mode_heterogeneity"]
        for metric in complexity_metrics:
            print(f"  {metric.replace('_', ' ').title():<28}", end="")
            for approach in approaches:
                value = successful_results[approach]["complexity_metrics"].get(metric, 0)
                print(f"{value:<15.3f}", end="")
            print()
        
        print("\nEXECUTION METRICS:")
        execution_metrics = ["serial_time", "critical_path_time", "parallel_efficiency"]
        for metric in execution_metrics:
            if metric in ["serial_time", "critical_path_time"]:
                continue  # Skip time metrics in table
            print(f"  {metric.replace('_', ' ').title():<28}", end="")
            for approach in approaches:
                value = successful_results[approach]["execution_metrics"].get(metric, 0)
                if value is not None:
                    print(f"{value:<15.3f}", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()
        
        print("-" * 100)


# Convenience functions for backward compatibility
def compare_dag_approaches(task: str, subtasks: Optional[List[SubTask]] = None) -> Dict[str, Any]:
    """Compare DAG building approaches."""
    framework = DAGComparison()
    return framework.compare_approaches(task, subtasks)


def save_comparison_results(results: Dict[str, Any], output_path: str):
    """Save comparison results."""
    framework = DAGComparison()
    framework.save_results(results, output_path)
