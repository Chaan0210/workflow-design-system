# tests/dag_comparison_test.py
import os
import sys
import pathlib
import itertools
import datetime
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_gaia
from dag_comparison import DAGComparison
from workflow_engine import WorkflowEngine
from visualizer import create_visualization
from models import Workflow, ComplexityMetrics, WorkflowQualityMetrics, ExecutionEstimate
from core import JsonSerializer


def create_dummy_metrics():
    """Create dummy metrics for visualization"""
    complexity_metrics = ComplexityMetrics(
        base_score=10.0,
        uncertainty_score=0.5,
        domain_complexity=5,
        coordination_complexity=5,
        computational_complexity=5,
        temporal_uncertainty=5,
        resource_uncertainty=5,
        outcome_uncertainty=5,
        critical_path_factor=0.8,
        parallel_efficiency=0.6,
        mode_heterogeneity=0.5,
        resource_conflict_factor=0.2,
        requires_replanning=False,
        risk_factors=[],
        mitigation_strategies=[]
    )
    
    quality_metrics = WorkflowQualityMetrics(
        completeness_score=0.9,
        coherence_score=0.85,
        efficiency_score=0.7,
        feasibility_score=0.95,
        overall_quality=0.85,
        validation_errors=[],
        warnings=[],
        suggestions=[]
    )
    
    execution_estimate = ExecutionEstimate(
        estimated_total_time=5.0,
        critical_path_time=3.0,
        parallel_time_savings=2.0,
        resource_requirements={"cpu_cores": 4, "memory_gb": 8},
        cost_estimate={"compute": 1.0, "storage": 0.2, "network": 0.1},
        bottlenecks=[]
    )
    
    return complexity_metrics, quality_metrics, execution_estimate


def visualize_comparison_results(results: dict, outdir: pathlib.Path, task_id: str, prompt: str):
    """Create visualizations for each successful approach"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    complexity_metrics, quality_metrics, execution_estimate = create_dummy_metrics()
    
    for approach_name, result in results.items():
        if approach_name == "comparison_summary" or not result.get("success", False):
            continue
            
        dag = result["dag"]
        subtasks = result.get("subtasks", [])
        
        # Create workflow object
        workflow = Workflow(
            task_id=f"{task_id}_{approach_name}",
            original_prompt=prompt,
            subtasks=subtasks,
            dag=dag,
        )
        
        # Create visualization
        output_path = outdir / f"{approach_name}_{timestamp}.png"
        try:
            create_visualization(workflow, complexity_metrics, quality_metrics, execution_estimate, output_path)
            print(f"✓ {approach_name.upper()} visualization: {output_path}")
        except Exception as e:
            print(f"✗ {approach_name.upper()} visualization failed: {e}")


def print_comparison_summary(results: dict, evaluation_report: dict = None):
    """Print a formatted summary of the comparison results"""
    print("\n" + "="*80)
    print("DAG BUILDING APPROACHES COMPARISON SUMMARY")
    print("="*80)
    
    summary = results.get("comparison_summary", {})
    successful = summary.get("successful_approaches", [])
    
    print(f"Successful approaches: {len(successful)}/3")
    for approach in successful:
        print(f"✓ {approach.upper()}")
    
    # Performance comparison
    if "performance_comparison" in summary:
        print("\n--- PERFORMANCE COMPARISON ---")
        print(f"{'Approach':<20} {'Build Time':<12} {'LLM Calls':<12} {'Nodes':<8} {'Edges':<8}")
        print("-" * 64)
        
        for approach, perf in summary["performance_comparison"].items():
            print(f"{approach:<20} {perf['build_time']:<12.3f} {perf['total_llm_calls']:<12} "
                  f"{perf['nodes']:<8} {perf['edges']:<8}")
    
    # Structural comparison
    if "structural_comparison" in summary:
        print("\n--- STRUCTURAL COMPARISON ---")
        print(f"{'Approach':<20} {'Is DAG':<8} {'Longest Path':<13} {'Avg Degree':<12} {'Density':<10}")
        print("-" * 68)
        
        for approach, struct in summary["structural_comparison"].items():
            is_dag = "✓" if struct.get("is_dag", False) else "✗"
            longest = struct.get("longest_path", "N/A")
            avg_deg = f"{struct.get('average_degree', 0):.2f}" if struct.get("average_degree") else "N/A"
            density = f"{struct.get('density', 0):.3f}" if struct.get("density") else "N/A"
            
            print(f"{approach:<20} {is_dag:<8} {longest:<13} {avg_deg:<12} {density:<10}")
    
    # Evaluation metrics if provided
    if evaluation_report:
        print("\n--- EVALUATION METRICS ---")
        per_approach = evaluation_report.get("per_approach", {})
        
        if per_approach:
            print(f"{'Approach':<20} {'Parallel Eff':<12} {'Edge F1':<10} {'Node Recall':<12}")
            print("-" * 58)
            
            for approach, metrics in per_approach.items():
                parallel_eff = f"{metrics.get('parallel_efficiency', 0):.3f}"
                edge_f1 = f"{metrics.get('edge_f1', 0):.3f}" if metrics.get('edge_f1') is not None else "N/A"
                node_recall = f"{metrics.get('node_recall', 0):.3f}" if metrics.get('node_recall') is not None else "N/A"
                
                print(f"{approach:<20} {parallel_eff:<12} {edge_f1:<10} {node_recall:<12}")
    
    print("\n" + "="*80)


def main():
    """Main comparison test function"""
    # Load environment
    dotenv_path = pathlib.Path(__file__).parent.parent / '.env'
    if not dotenv_path.exists():
        sys.exit(f"❗ .env file not found at {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path)
    
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("❗ OPENAI_API_KEY not found")
    
    # Setup output directory
    outdir = pathlib.Path(__file__).parent.parent / "workflow_outputs"
    outdir.mkdir(exist_ok=True)
    
    # Load test task
    task_id, prompt = next(itertools.islice(load_gaia(), 1))
    print(f"Testing task: {task_id}")
    print(f"Prompt: {prompt}")
    
    # Create comparison framework
    comparison = DAGComparison()
    
    # Run comparison
    print("\n🔄 Starting DAG approach comparison...")
    results = comparison.compare_approaches(prompt)
    
    # Run evaluation
    print("\n📊 Evaluating workflows...")
    evaluation_report = comparison.evaluate_results(results)
    
    # Print comprehensive summary
    print_comparison_summary(results, evaluation_report)
    
    # Save results with evaluation metrics
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = outdir / f"results_{timestamp}.json"
    
    # Combine comparison and evaluation results
    comprehensive_results = {
        "comparison_results": results,
        "evaluation": evaluation_report,
        "metadata": {
            "task_id": task_id,
            "prompt": prompt,
            "timestamp": timestamp
        }
    }
    
    # Save comprehensive results
    JsonSerializer.save_json(comprehensive_results, str(results_path))
    print(f"\n📊 Results saved: {results_path}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    visualize_comparison_results(results, outdir, task_id, prompt)
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
