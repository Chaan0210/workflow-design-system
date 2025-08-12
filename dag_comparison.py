# dag_comparison.py
import time
from typing import Dict, Any, List, Optional, Tuple

from models import SubTask
from workflow_engine import WorkflowEngine
from core import JsonSerializer, ComparisonEvaluator
from core.evaluation_utils import MetricsCalculator, QualityAssessor
from core.validation_utils import WorkflowValidator

from approaches.bidirectional import BidirectionalApproach
from approaches.hierarchical import HierarchicalApproach  
from approaches.matrix import MatrixApproach
from approaches.bidirectional_parallel import BidirectionalParallelApproach

class DAGComparison:
    """Main DAG comparison framework with optimization support."""
    
    def __init__(self, include_parallel_approaches: bool = True):
        # Only use parallel approaches for bidirectional and matrix, keep hierarchical as-is
        self.approaches = {
            "hierarchical": HierarchicalApproach(),
        }
        
        # Use only parallel versions of bidirectional and matrix
        if include_parallel_approaches:
            self.approaches.update({
                "bidirectional_parallel": BidirectionalParallelApproach(),
                "matrix": MatrixApproach()
            })
        else:
            # Fallback to original approaches if parallel is disabled
            self.approaches.update({
                "bidirectional": BidirectionalApproach(),
                "matrix": MatrixApproach()
            })
        
        self.evaluator = ComparisonEvaluator()
        self.workflow_validator = WorkflowValidator()
    
    def compare_approaches(self, task: str, subtasks: Optional[List[SubTask]] = None, 
                         approaches_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare DAG building approaches with optional filtering."""
        
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
        
        # 접근법 필터링
        active_approaches = self.approaches
        if approaches_filter:
            active_approaches = {
                name: approach for name, approach in self.approaches.items() 
                if name in approaches_filter
            }
            print(f"\n🔍 Filtering to approaches: {list(active_approaches.keys())}")
        
        print(f"\n🔄 Comparing DAG approaches for {len(subtasks)} subtasks...")
        print(f"   Active approaches: {list(active_approaches.keys())}")
        
        results = {}
        
        # Test each approach
        for name, approach in active_approaches.items():
            print(f"\n{'='*60}")
            print(f"🚀 Running {name.upper()} approach...")
            print('='*60)
            
            try:
                dag, metrics = approach.build_dag(subtasks, task)
                
                # Evaluate workflow quality and structure with LLM support
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
                    
                    # 병렬 최적화 관련 메트릭 출력
                    if metrics.get('parallel_optimization'):
                        print(f"   🚀 Parallel optimization: ENABLED")
                        if 'analysis_time' in metrics:
                            print(f"   ⚡ Analysis time: {metrics['analysis_time']:.2f}s")
                        if 'pairs_per_second' in metrics:
                            print(f"   📊 Analysis rate: {metrics['pairs_per_second']:.1f} pairs/sec")
                        if 'parallel_efficiency_block' in metrics and 'parallel_efficiency_path' in metrics:
                            print(f"   🔀 Block efficiency: {metrics['parallel_efficiency_block']:.3f}")
                            print(f"   🔀 Path efficiency: {metrics['parallel_efficiency_path']:.3f}")
                        elif 'parallel_efficiency' in metrics:
                            print(f"   🔀 Parallel efficiency: {metrics['parallel_efficiency']:.3f}")
                    
                    if 'fallback_used' in metrics:
                        print(f"   Used fallback: {metrics['fallback_used']}")
                    
                    # Print workflow evaluation results
                    eval_results = evaluation_results
                    print(f"   📊 Workflow Evaluation:")
                    
                    def format_score(score):
                        return f"{score:.3f}" if score is not None else "null"
                    
                    print(f"      Completeness: {format_score(eval_results['quality_metrics']['completeness_score'])}")
                    print(f"      Coherence: {format_score(eval_results['quality_metrics']['coherence_score'])}")
                    print(f"      Efficiency: {format_score(eval_results['quality_metrics']['efficiency_score'])}")
                    print(f"      Feasibility: {format_score(eval_results['quality_metrics']['feasibility_score'])}")
                    
                    # Print structural metrics
                    struct_metrics = eval_results['structural_metrics']
                    print(f"   🏗️  Structural Metrics:")
                    print(f"      Nodes: {struct_metrics['num_nodes']}, Edges: {struct_metrics['num_edges']}")
                    print(f"      Density: {struct_metrics['density']:.3f}")
                    print(f"      Average Degree: {struct_metrics['average_degree']:.3f}")
                    print(f"      Parallel Efficiency: {struct_metrics['parallel_efficiency']:.3f}")
                    
                    # Print complexity metrics
                    complexity = eval_results['complexity_metrics']
                    print(f"   🧠 Complexity Analysis:")
                    print(f"      Domain: {complexity['domain_complexity']:.1f}/10")
                    print(f"      Coordination: {complexity['coordination_complexity']:.1f}/10")
                    print(f"      Computational: {complexity['computational_complexity']:.1f}/10")
                    
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
    
    def compare_optimization_impact(self, task: str, subtasks: Optional[List[SubTask]] = None) -> Dict[str, Any]:
        """최적화 전후 성능 비교"""
        print("\n🔬 OPTIMIZATION IMPACT ANALYSIS")
        print("="*80)
        
        # Note: This method now compares current parallel approaches with legacy versions
        # Create temporary instances for comparison
        legacy_approaches = {
            "bidirectional": BidirectionalApproach(),
            "matrix": MatrixApproach()
        }
        
        # Temporarily add legacy approaches for comparison
        original_approaches_dict = self.approaches.copy()
        self.approaches.update(legacy_approaches)
        
        original_approaches = ["bidirectional", "matrix"]
        optimized_approaches = ["bidirectional_parallel", "matrix"]
        
        comparison_results = self.compare_approaches(
            task, subtasks, 
            approaches_filter=original_approaches + optimized_approaches
        )
        
        # Restore original approaches
        self.approaches = original_approaches_dict
        
        # 최적화 효과 분석
        optimization_analysis = {}
        
        for original, optimized in zip(original_approaches, optimized_approaches):
            if original in comparison_results and optimized in comparison_results:
                orig_metrics = comparison_results[original]["metrics"]
                opt_metrics = comparison_results[optimized]["metrics"]
                
                time_improvement = self._calculate_improvement(
                    orig_metrics.get("build_time", 0),
                    opt_metrics.get("build_time", 0)
                )
                
                llm_efficiency = self._calculate_llm_efficiency(
                    orig_metrics.get("total_llm_calls", 0),
                    orig_metrics.get("build_time", 1),
                    opt_metrics.get("total_llm_calls", 0),
                    opt_metrics.get("build_time", 1)
                )
                
                optimization_analysis[f"{original}_vs_{optimized}"] = {
                    "time_improvement_percent": time_improvement,
                    "llm_call_efficiency": llm_efficiency,
                    "original_time": orig_metrics.get("build_time", 0),
                    "optimized_time": opt_metrics.get("build_time", 0),
                    "original_llm_calls": orig_metrics.get("total_llm_calls", 0),
                    "optimized_llm_calls": opt_metrics.get("total_llm_calls", 0)
                }
        
        comparison_results["optimization_analysis"] = optimization_analysis
        self._print_optimization_summary(optimization_analysis)
        
        return comparison_results
    
    def _calculate_improvement(self, original: float, optimized: float) -> float:
        """개선 비율 계산"""
        if original <= 0:
            return 0.0
        return ((original - optimized) / original) * 100
    
    def _calculate_llm_efficiency(self, orig_calls: int, orig_time: float, 
                                opt_calls: int, opt_time: float) -> Dict[str, float]:
        """LLM 호출 효율성 계산"""
        orig_rate = orig_calls / orig_time if orig_time > 0 else 0
        opt_rate = opt_calls / opt_time if opt_time > 0 else 0
        
        return {
            "original_calls_per_second": orig_rate,
            "optimized_calls_per_second": opt_rate,
            "efficiency_improvement": ((opt_rate - orig_rate) / orig_rate * 100) if orig_rate > 0 else 0
        }
    
    def _print_optimization_summary(self, optimization_analysis: Dict[str, Any]):
        """최적화 요약 출력"""
        print("\n📊 OPTIMIZATION SUMMARY")
        print("-" * 80)
        
        for comparison, data in optimization_analysis.items():
            print(f"\n{comparison.upper().replace('_', ' → ')}:")
            print(f"  ⏱️  Time improvement: {data['time_improvement_percent']:.1f}%")
            print(f"      ({data['original_time']:.2f}s → {data['optimized_time']:.2f}s)")
            
            efficiency = data['llm_call_efficiency']
            print(f"  🚀 LLM efficiency improvement: {efficiency['efficiency_improvement']:.1f}%")
            print(f"      ({efficiency['original_calls_per_second']:.1f} → {efficiency['optimized_calls_per_second']:.1f} calls/sec)")
            
            print(f"  📞 LLM calls: {data['original_llm_calls']} → {data['optimized_llm_calls']}")
        
        print("-" * 80)
    
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
            if approach_name in ["comparison_summary", "evaluation_comparison", "optimization_analysis"]:
                continue
            
            if result.get("success", False):
                summary["successful_approaches"].append(approach_name)
                metrics = result["metrics"]
                
                # Use consistent node/edge counting (exclude group nodes)
                dag = result["dag"]
                actual_nodes = [n for n in dag.nodes() if not dag.nodes[n].get('is_group', False)]
                actual_dag = dag.subgraph(actual_nodes).copy()
                
                summary["performance_comparison"][approach_name] = {
                    "build_time": metrics["build_time"],
                    "total_llm_calls": metrics.get("total_llm_calls", 0),
                    "nodes": len(actual_nodes),
                    "edges": actual_dag.number_of_edges(),
                    "parallel_optimization": metrics.get("parallel_optimization", False)
                }
                
                # Structural analysis (use same filtered DAG for consistency)
                if nx.is_directed_acyclic_graph(actual_dag) and len(actual_nodes) > 0:
                    try:
                        summary["structural_comparison"][approach_name] = {
                            "is_dag": True,
                            "longest_path": len(nx.dag_longest_path(actual_dag)),
                            "average_degree": sum(dict(actual_dag.degree()).values()) / len(actual_nodes),
                            "density": nx.density(actual_dag)
                        }
                    except:
                        summary["structural_comparison"][approach_name] = {"is_dag": True, "analysis_failed": True}
                else:
                    summary["structural_comparison"][approach_name] = {"is_dag": False}
        
        return summary
    
    def _evaluate_approach_workflow(self, dag, subtasks: List[SubTask], task: str, 
                                   build_metrics: Dict[str, Any], approach_name: str) -> Dict[str, Any]:
        """Simplified workflow evaluation for a single approach."""
        print(f"      🔍 Evaluating {approach_name} workflow...")
        
        # Try to get LLM client for quality analysis
        llm_client = None
        try:
            from core import LLMClient
            llm_client = LLMClient()
        except Exception as e:
            print(f"      ⚠️  LLM client not available, using heuristic fallback: {e}")
        
        # 1. Structural metrics (simplified)
        structural_metrics = MetricsCalculator.structural_metrics(dag)
        
        # 2. Quality assessment with LLM support
        quality_metrics = QualityAssessor.assess_workflow_quality(
            {"subtasks": subtasks, "dag": dag}, 
            subtasks, 
            dag,
            llm_client=llm_client,
            main_task=task
        )
        
        # 3. Complexity metrics with LLM support
        complexity_metrics = MetricsCalculator.complexity_metrics(
            {"subtasks": subtasks, "dag": dag},
            llm_client=llm_client,
            main_task=task
        )
        
        # 4. Basic validation (keep minimal validation)
        validation_issues = self.workflow_validator.validate_workflow_structure({
            "subtasks": subtasks,
            "dag": dag
        })
        
        return {
            "structural_metrics": structural_metrics,
            "quality_metrics": quality_metrics,
            "complexity_metrics": complexity_metrics,
            "validation_issues": validation_issues,
            "approach_name": approach_name,
            "build_metrics": build_metrics
        }
    
    def _generate_evaluation_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation comparison across approaches."""
        comparison = {
            "quality_comparison": {},
            "complexity_comparison": {},
            "structural_comparison": {}
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
            
            comparison["quality_comparison"][approach_name] = quality
            comparison["complexity_comparison"][approach_name] = complexity
            comparison["structural_comparison"][approach_name] = structural
        
        
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
        quality_metrics = ["completeness_score", "coherence_score", "efficiency_score", "feasibility_score"]
        for metric in quality_metrics:
            print(f"  {metric.replace('_', ' ').title():<28}", end="")
            for approach in approaches:
                value = successful_results[approach]["quality_metrics"].get(metric, 0)
                if value is not None:
                    print(f"{value:<15.3f}", end="")
                else:
                    print(f"{'null':<15}", end="")
            print()
        
        print("\nSTRUCTURAL METRICS:")
        structural_metrics = ["num_nodes", "num_edges", "density", "average_degree", "parallel_efficiency"]
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
        complexity_metrics = ["domain_complexity", "coordination_complexity", "computational_complexity"]
        for metric in complexity_metrics:
            print(f"  {metric.replace('_', ' ').title():<28}", end="")
            for approach in approaches:
                value = successful_results[approach]["complexity_metrics"].get(metric, 0)
                print(f"{value:<15.3f}", end="")
            print()
        
        # Execution metrics are now included in structural metrics
        
        print("-" * 100)


# Convenience functions for backward compatibility
def compare_dag_approaches(task: str, subtasks: Optional[List[SubTask]] = None) -> Dict[str, Any]:
    """Compare DAG building approaches (now uses parallel versions by default)."""
    framework = DAGComparison()
    return framework.compare_approaches(task, subtasks)


def compare_optimization_impact(task: str, subtasks: Optional[List[SubTask]] = None) -> Dict[str, Any]:
    """Compare optimization impact between original and parallel approaches."""
    framework = DAGComparison()
    return framework.compare_optimization_impact(task, subtasks)


def save_comparison_results(results: Dict[str, Any], output_path: str):
    """Save comparison results."""
    framework = DAGComparison()
    framework.save_results(results, output_path)
