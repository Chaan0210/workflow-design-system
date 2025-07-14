# main.py
import os
import sys
import argparse
import pathlib
import itertools
from tqdm import tqdm

from utils import load_gaia, save_json
from workflow_planner import plan_workflow
from visualizer import create_visualization, create_metrics_report


def main():
    ap = argparse.ArgumentParser(description="GAIA Workflow Planner")
    ap.add_argument("--limit", type=int, default=5, help="Max tasks to process")
    ap.add_argument("--viz", action="store_true", help="Save visualizations and reports")
    ap.add_argument("--detailed", action="store_true", help="Generate detailed analysis reports")
    ap.add_argument("--replan-threshold", type=float, default=20.0, help="Complexity threshold for re-planning")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("❗ OPENAI_API_KEY 환경변수를 먼저 설정하세요.")

    outdir = pathlib.Path("workflow_outputs")
    outdir.mkdir(exist_ok=True)
    
    print(f"🚀 Starting Workflow Analysis")
    print(f"   Input: Hugging Face GAIA Dataset (gaia-benchmark/GAIA)")
    print(f"   Output: {outdir.resolve()}")
    print(f"   Tasks to process: {args.limit}")
    print(f"   Visualization: {'✅' if args.viz else '❌'}")
    print(f"   Detailed reports: {'✅' if args.detailed else '❌'}")
    print("-" * 60)

    successful_plans = 0
    replanning_needed = 0
    
    for task_id, prompt in tqdm(itertools.islice(load_gaia(), args.limit), 
                              desc="Analyzing workflows", unit="task"):
        try:
            # Workflow planning
            wf, complexity_metrics, quality_metrics, execution_estimate, issues = plan_workflow(task_id, prompt)
            
            # Save JSON report
            save_json(wf.report, outdir / f"{task_id}.json")
            
            # Generate visualizations and detailed reports
            if args.viz or args.detailed:
                create_visualization(wf, complexity_metrics, quality_metrics, 
                                           execution_estimate, outdir / f"{task_id}.png")
                
                if args.detailed:
                    create_metrics_report(wf, complexity_metrics, quality_metrics, 
                                        execution_estimate, issues, outdir / f"{task_id}")
            
            # Track statistics
            successful_plans += 1
            if complexity_metrics.requires_replanning:
                replanning_needed += 1
                
            # Print summary for this task
            print(f"\n📋 Task {task_id} Summary:")
            print(f"   Quality: {quality_metrics.overall_quality:.3f}")
            print(f"   Complexity: {complexity_metrics.base_score:.2f}")
            print(f"   Est. Time: {execution_estimate.estimated_total_time:.1f}h")
            if complexity_metrics.requires_replanning:
                print(f"   ⚠️ Re-planning recommended")
                
        except Exception as e:
            print(f"❌ Error processing task {task_id}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print(f"📊 WORKFLOW ANALYSIS COMPLETE")
    print(f"   Successfully analyzed: {successful_plans}/{args.limit} tasks")
    print(f"   Re-planning needed: {replanning_needed} tasks ({100*replanning_needed/max(successful_plans,1):.1f}%)")
    print(f"   Results saved to: {outdir.resolve()}")
    
    if args.viz:
        print(f"   📈 Visualizations: {successful_plans} files")
    if args.detailed:
        print(f"   📄 Detailed reports: {successful_plans} files")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
