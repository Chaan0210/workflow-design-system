# main.py
import os
import sys
import argparse
import pathlib
import itertools
from tqdm import tqdm

from utils import load_gaia
from workflow_engine import WorkflowEngine
from dag_comparison import DAGComparison
from core import JsonSerializer
from visualizer import create_visualization


def main():
    """Main function using the simplified workflow engine."""
    ap = argparse.ArgumentParser(description="Simplified GAIA Workflow Planner")
    ap.add_argument("--limit", type=int, default=5, help="Max tasks to process")
    ap.add_argument("--viz", action="store_true", help="Save visualizations")
    ap.add_argument("--comparison", action="store_true", help="Run DAG approach comparison")
    ap.add_argument("--mode", choices=["workflow", "comparison"], default="workflow",
                   help="Run mode: single workflow or comparison")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("❗ OPENAI_API_KEY 환경변수를 먼저 설정하세요.")

    outdir = pathlib.Path("workflow_outputs")
    outdir.mkdir(exist_ok=True)
    
    print(f"🚀 Starting Simplified Workflow Analysis")
    print(f"   Mode: {args.mode}")
    print(f"   Output: {outdir.resolve()}")
    print(f"   Tasks to process: {args.limit}")
    print(f"   Visualization: {'✅' if args.viz else '❌'}")
    print("-" * 60)

    if args.mode == "workflow":
        run_workflow_mode(args, outdir)
    else:
        run_comparison_mode(args, outdir)


def run_workflow_mode(args, outdir):
    """Run single workflow planning mode."""
    engine = WorkflowEngine()
    
    successful_plans = 0
    replanning_needed = 0
    
    for task_id, prompt in tqdm(itertools.islice(load_gaia(), args.limit), 
                              desc="Analyzing workflows", unit="task"):
        try:
            # Plan workflow using unified engine
            workflow, complexity_metrics, quality_metrics, execution_estimate, issues = engine.plan_workflow(task_id, prompt)
            
            # Save JSON report
            JsonSerializer.save_json(workflow.report, str(outdir / f"{task_id}.json"))
            
            # Generate visualization if requested
            if args.viz:
                viz_path = outdir / f"{task_id}.png"
                try:
                    create_visualization(workflow, complexity_metrics, quality_metrics, 
                                       execution_estimate, viz_path)
                    print(f"   📈 Visualization saved: {viz_path}")
                except Exception as e:
                    print(f"   ❌ Visualization failed: {e}")
            
            # Track statistics
            successful_plans += 1
            if complexity_metrics.requires_replanning:
                replanning_needed += 1
                
            # Print summary
            print(f"\n📋 Task {task_id} Summary:")
            print(f"   Quality: {quality_metrics.overall_quality:.3f}")
            print(f"   Complexity: {complexity_metrics.base_score:.2f}")
            print(f"   Est. Time: {execution_estimate.estimated_total_time:.1f}h")
            if complexity_metrics.requires_replanning:
                print(f"   ⚠️ Re-planning recommended")
                
        except Exception as e:
            print(f"❌ Error processing task {task_id}: {str(e)}")
            continue
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"📊 WORKFLOW ANALYSIS COMPLETE")
    print(f"   Successfully analyzed: {successful_plans}/{args.limit} tasks")
    print(f"   Re-planning needed: {replanning_needed} tasks")
    print(f"   Results saved to: {outdir.resolve()}")
    print("=" * 60)


def run_comparison_mode(args, outdir):
    """Run DAG approach comparison mode."""
    comparison_framework = DAGComparison()
    
    successful_comparisons = 0
    
    for task_id, prompt in tqdm(itertools.islice(load_gaia(), args.limit),
                              desc="Comparing approaches", unit="task"):
        try:
            # Run comparison
            print(f"\n🔄 Comparing approaches for task: {task_id}")
            results = comparison_framework.compare_approaches(prompt)
            
            # Evaluate results
            evaluation = comparison_framework.evaluate_results(results)
            
            # Combine results
            comprehensive_results = {
                "comparison_results": results,
                "evaluation": evaluation,
                "metadata": {
                    "task_id": task_id,
                    "prompt": prompt
                }
            }
            
            # Save results
            results_path = outdir / f"comparison_{task_id}.json"
            comparison_framework.save_results(comprehensive_results, str(results_path))
            
            # Print summary
            summary = results.get("comparison_summary", {})
            successful_approaches = summary.get("successful_approaches", [])
            
            print(f"✅ Comparison complete for {task_id}")
            print(f"   Successful approaches: {len(successful_approaches)}/3")
            for approach in successful_approaches:
                print(f"   ✓ {approach.upper()}")
            
            # Generate visualizations if requested
            if args.viz:
                for approach_name, result in results.items():
                    if approach_name != "comparison_summary" and result.get("success", False):
                        try:
                            # Create mock workflow for visualization
                            from models import Workflow
                            mock_workflow = Workflow(
                                task_id=f"{task_id}_{approach_name}",
                                original_prompt=prompt,
                                subtasks=result.get("subtasks", []),
                                dag=result["dag"]
                            )
                            
                            # Create dummy metrics for visualization
                            from models import ComplexityMetrics, WorkflowQualityMetrics, ExecutionEstimate
                            dummy_complexity = ComplexityMetrics(
                                base_score=10.0, uncertainty_score=0.5, domain_complexity=5,
                                coordination_complexity=5, computational_complexity=5,
                                temporal_uncertainty=5, resource_uncertainty=5, outcome_uncertainty=5,
                                critical_path_factor=0.8, parallel_efficiency=0.6, mode_heterogeneity=0.5,
                                resource_conflict_factor=0.2, requires_replanning=False,
                                risk_factors=[], mitigation_strategies=[]
                            )
                            dummy_quality = WorkflowQualityMetrics(
                                completeness_score=0.9, coherence_score=0.85, efficiency_score=0.7,
                                feasibility_score=0.95, overall_quality=0.85,
                                validation_errors=[], warnings=[], suggestions=[]
                            )
                            dummy_execution = ExecutionEstimate(
                                estimated_total_time=5.0, critical_path_time=3.0, parallel_time_savings=2.0,
                                resource_requirements={"cpu_cores": 4, "memory_gb": 8},
                                cost_estimate={"compute": 1.0, "storage": 0.2, "network": 0.1},
                                bottlenecks=[]
                            )
                            
                            viz_path = outdir / f"comparison_{task_id}_{approach_name}.png"
                            create_visualization(mock_workflow, dummy_complexity, dummy_quality, 
                                               dummy_execution, viz_path)
                            
                        except Exception as e:
                            print(f"   ❌ Visualization failed for {approach_name}: {e}")
            
            successful_comparisons += 1
            
        except Exception as e:
            print(f"❌ Error in comparison for task {task_id}: {str(e)}")
            continue
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"📊 DAG COMPARISON COMPLETE")
    print(f"   Successfully compared: {successful_comparisons}/{args.limit} tasks")
    print(f"   Results saved to: {outdir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()