# tests/gaia_dag_test.py
import os
import sys
import pathlib
import itertools
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_gaia
from workflow_engine import WorkflowEngine
from visualizer import create_visualization
from core import JsonSerializer


def main():
    """Test workflow engine with GAIA dataset"""
    
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
    print(f"🧪 Testing GAIA DAG generation")
    print(f"Task ID: {task_id}")
    print(f"Prompt: {prompt}")
    print("-" * 80)
    
    # Create workflow engine
    engine = WorkflowEngine()
    
    try:
        # Plan workflow
        print("🔍 Planning workflow...")
        workflow, complexity_metrics, quality_metrics, execution_estimate, issues = engine.plan_workflow(task_id, prompt)
        
        print(f"✅ Workflow planning completed!")
        print(f"   Sub-tasks: {len(workflow.subtasks)}")
        print(f"   Dependencies: {workflow.dag.number_of_edges()}")
        print(f"   Quality Score: {quality_metrics.overall_quality:.3f}")
        print(f"   Complexity: {complexity_metrics.base_score:.2f}")
        print(f"   Estimated Time: {execution_estimate.estimated_total_time:.1f}h")
        
        # Save results
        output_path = outdir / f"gaia_dag_test_{task_id}.json"
        JsonSerializer.save_json(workflow.report, str(output_path))
        print(f"📊 Results saved: {output_path}")
        
        # Create visualization
        viz_path = outdir / f"gaia_dag_test_{task_id}.png"
        try:
            create_visualization(workflow, complexity_metrics, quality_metrics, execution_estimate, viz_path)
            print(f"📈 Visualization saved: {viz_path}")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
        
        # Print summary
        print("\n" + "="*80)
        print("WORKFLOW SUMMARY")
        print("="*80)
        
        print(f"Task ID: {workflow.task_id}")
        print(f"Quality: {quality_metrics.overall_quality:.3f}/1.0")
        print(f"Complexity: {complexity_metrics.base_score:.2f}")
        print(f"Uncertainty: {complexity_metrics.uncertainty_score:.3f}")
        print(f"Parallel Efficiency: {complexity_metrics.parallel_efficiency:.3f}")
        print(f"Estimated Time: {execution_estimate.estimated_total_time:.1f}h")
        print(f"Time Savings: {execution_estimate.parallel_time_savings:.1f}h")
        
        if complexity_metrics.requires_replanning:
            print("⚠️ Re-planning recommended")
        
        # Print sub-tasks
        print(f"\nSub-tasks ({len(workflow.subtasks)}):")
        for st in workflow.subtasks:
            print(f"  {st.id} ({st.mode}): {st.description}")
        
        # Print dependencies
        if workflow.dag.number_of_edges() > 0:
            print(f"\nDependencies ({workflow.dag.number_of_edges()}):")
            for u, v in workflow.dag.edges():
                conf = workflow.dag[u][v].get('confidence', 'N/A')
                print(f"  {u} → {v} (confidence: {conf})")
        
        # Print issues if any
        critical_issues = issues.get('critical_issues', [])
        if critical_issues:
            print(f"\n⚠️ Critical Issues:")
            for issue in critical_issues:
                print(f"  - {issue}")
        
        recommendations = issues.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  - {rec}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
