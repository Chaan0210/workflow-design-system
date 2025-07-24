# tests/gaia_dag_test.py
import os
import sys
import pathlib
import itertools
import datetime
from dotenv import load_dotenv

# Add project root to sys.path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_gaia, save_json
from decomposition import decompose
from dag_builder import build_dag
from visualizer import create_visualization
from models import Workflow, ComplexityMetrics, WorkflowQualityMetrics, ExecutionEstimate

def main():
    # Load environment variables from .env file
    dotenv_path = pathlib.Path(__file__).parent.parent / '.env'
    if not dotenv_path.exists():
        sys.exit(f"❗ .env file not found at {dotenv_path}. Please create it and add your OPENAI_API_KEY.")
    load_dotenv(dotenv_path=dotenv_path)

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("❗ OPENAI_API_KEY not found in .env file or environment variables.")

    # Output directory should be in the project root, not the test directory
    outdir = pathlib.Path(__file__).parent.parent / "workflow_outputs"
    outdir.mkdir(exist_ok=True)

    # Load the first task from the GAIA dataset
    task_id, prompt = next(itertools.islice(load_gaia(), 1))

    print(f"Processing task: {task_id}")
    print(f"Prompt: {prompt}")

    # 1. Decompose the task into subtasks
    subtasks = decompose(prompt)
    print(f"Decomposed into {len(subtasks)} subtasks.")

    # 2. Build the DAG
    dag = build_dag(subtasks)
    print(f"DAG built with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges.")

    # Create a dummy workflow object for visualization
    workflow = Workflow(
        task_id=task_id,
        original_prompt=prompt,
        subtasks=subtasks,
        dag=dag,
    )
    
    # Create dummy metrics for visualization
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
    )
    quality_metrics = WorkflowQualityMetrics(
        completeness_score=0.9,
        coherence_score=0.85,
        efficiency_score=0.7,
        feasibility_score=0.95,
        overall_quality=0.85,
    )
    execution_estimate = ExecutionEstimate(
        estimated_total_time=5.0,
        critical_path_time=3.0,
        parallel_time_savings=2.0,
        resource_requirements={"cpu_cores": 4, "memory_gb": 8},
        cost_estimate={"compute": 1.0, "storage": 0.2, "network": 0.1},
    )

    # 3. Visualize the DAG
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = outdir / f"gaia_dag_test_{timestamp}.png"
    create_visualization(workflow, complexity_metrics, quality_metrics, execution_estimate, output_path)
    print(f"DAG visualization saved to {output_path}")

if __name__ == "__main__":
    main()
