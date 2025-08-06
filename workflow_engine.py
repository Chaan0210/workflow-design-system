# workflow_engine.py
import json
import time
import itertools
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import networkx as nx

from models import Workflow, SubTask, ComplexityMetrics, WorkflowQualityMetrics, ExecutionEstimate
from core import (
    LLMClient, PromptManager, DAGProcessor, MetricsCalculator, 
    QualityAssessor, WorkflowValidator, DAGValidator, JsonSerializer
)


class WorkflowEngine:
    """Unified workflow engine for planning, building, and evaluating workflows."""
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm_client = llm_client or LLMClient()
        self.dag_processor = DAGProcessor()
        self.validator = WorkflowValidator(self.llm_client)
        
        # Constants for DAG building
        self.LOW_CONF_THRESHOLD = 0.4
        self.CUTOFF_THRESHOLD = 0.5
    
    def plan_workflow(self, task_id: str, prompt: str) -> Tuple[Workflow, ComplexityMetrics, WorkflowQualityMetrics, ExecutionEstimate, Dict]:
        """Complete workflow planning pipeline."""
        
        print(f"🔍 Planning workflow for task: {task_id}")
        
        # Create workflow object
        workflow = Workflow(task_id=task_id, original_prompt=prompt)
        
        # 1. Decompose task into subtasks
        print("  📋 Decomposing into sub-tasks...")
        workflow.subtasks = self._decompose_task(prompt)
        print(f"     Generated {len(workflow.subtasks)} sub-tasks")
        
        # 2. Build DAG
        print("  🔗 Building dependency graph...")
        workflow.dag = self._build_dag(workflow.subtasks, prompt)
        print(f"     Created DAG with {workflow.dag.number_of_edges()} dependencies")
        
        # 3. Classify modes
        print("  🎯 Analyzing task modes...")
        self._classify_modes(workflow.subtasks, prompt)
        wide_count = len([st for st in workflow.subtasks if st.mode == "WIDE"])
        deep_count = len([st for st in workflow.subtasks if st.mode == "DEEP"])
        print(f"     Classified: {wide_count} WIDE, {deep_count} DEEP tasks")
        
        # 4. Calculate complexity
        print("  📊 Computing complexity metrics...")
        complexity_metrics = self._calculate_complexity(workflow)
        print(f"     Complexity: {complexity_metrics.base_score:.2f}, Uncertainty: {complexity_metrics.uncertainty_score:.3f}")
        
        # 5. Validate quality
        print("  ✅ Validating workflow quality...")
        quality_metrics = self._validate_quality(workflow)
        print(f"     Quality score: {quality_metrics.overall_quality:.3f}/1.0")
        
        # 6. Estimate execution
        print("  ⏱️ Estimating execution metrics...")
        execution_estimate = self._estimate_execution(workflow)
        print(f"     Estimated time: {execution_estimate.estimated_total_time:.1f}h")
        
        # 7. Diagnose issues
        print("  🔍 Diagnosing potential issues...")
        issues = self._diagnose_issues(workflow, quality_metrics, complexity_metrics)
        
        # 8. Generate report
        workflow.report = self._generate_report(workflow, complexity_metrics, quality_metrics, execution_estimate, issues)
        
        return workflow, complexity_metrics, quality_metrics, execution_estimate, issues
    
    def _decompose_task(self, task: str) -> List[SubTask]:
        """Decompose task into subtasks."""
        prompt = PromptManager.format_decomposition_prompt(task)
        
        try:
            json_response = self.llm_client.call_text(
                prompt, 
                system="You are a helpful data-only JSON generator. Respond with nothing but valid JSON.",
                temperature=0
            )
            
            # Clean markdown if present
            if json_response.startswith("```json"):
                json_response = json_response[7:-4]
            
            items = json.loads(json_response)
            return [SubTask(id=item["id"], description=item["desc"]) for item in items]
            
        except Exception as e:
            print(f"Decomposition failed, using fallback: {str(e)}")
            # Fallback: create simple sequential subtasks
            return [
                SubTask(id="S1", description="Analyze the task requirements"),
                SubTask(id="S2", description="Gather necessary information"),
                SubTask(id="S3", description="Process and analyze data"),
                SubTask(id="S4", description="Generate solution"),
                SubTask(id="S5", description="Format and present results")
            ]
    
    def _build_dag(self, subtasks: List[SubTask], original_task: str = "") -> nx.DiGraph:
        """Build DAG using bidirectional analysis."""
        dag = nx.DiGraph()
        
        # Add nodes
        for st in subtasks:
            dag.add_node(st.id, obj=st, description=st.description)
        
        edge_conf_map: Dict[Tuple[str, str], float] = {}
        low_conf_candidates: List[Tuple[str, str, float]] = []
        
        # Bidirectional dependency analysis
        for a, b in itertools.combinations(subtasks, 2):
            dep_ab, conf_ab = self._ask_dependency(a, b, original_task)
            dep_ba, conf_ba = self._ask_dependency(b, a, original_task)
            
            # Resolve conflicts
            if dep_ab and dep_ba:
                if conf_ab >= conf_ba:
                    dep_ba = False
                else:
                    dep_ab = False
            
            # Process AB edge
            if dep_ab:
                has_conflict, shared = self._ask_resource_conflict(a, b, original_task)
                if conf_ab >= self.CUTOFF_THRESHOLD:
                    dag.add_edge(a.id, b.id,
                               confidence=conf_ab,
                               has_resource_conflict=has_conflict,
                               shared_resources=shared)
                    edge_conf_map[(a.id, b.id)] = conf_ab
                elif conf_ab >= self.LOW_CONF_THRESHOLD:
                    low_conf_candidates.append((a.id, b.id, conf_ab))
            
            # Process BA edge
            if dep_ba:
                has_conflict, shared = self._ask_resource_conflict(b, a, original_task)
                if conf_ba >= self.CUTOFF_THRESHOLD:
                    dag.add_edge(b.id, a.id,
                               confidence=conf_ba,
                               has_resource_conflict=has_conflict,
                               shared_resources=shared)
                    edge_conf_map[(b.id, a.id)] = conf_ba
                elif conf_ba >= self.LOW_CONF_THRESHOLD:
                    low_conf_candidates.append((b.id, a.id, conf_ba))
        
        # Add low confidence edges that don't create conflicts
        for u, v, c in sorted(low_conf_candidates, key=lambda x: -x[2]):
            if not dag.has_edge(u, v) and not dag.has_edge(v, u):
                dag.add_edge(u, v, confidence=round(c, 3),
                           has_resource_conflict=False, shared_resources=[])
        
        # Remove cycles
        dag = self.dag_processor.resolve_cycles_intelligently(dag, edge_conf_map)
        
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Graph contains cycles after resolution")
        
        # Post-process
        self.dag_processor.post_process_graph(dag)
        
        # Apply rule-based pruning
        dag = self.dag_processor.apply_rule_based_pruning(dag)
        
        return dag
    
    def _ask_dependency(self, a: SubTask, b: SubTask, original_task: str = "") -> Tuple[bool, float]:
        """Check dependency between two tasks."""
        prompt = PromptManager.format_dependency_prompt(original_task, a.description, b.description)
        
        try:
            data = self.llm_client.call_json(prompt, validator=DAGValidator.validate_dependency_json)
            return bool(data["dependent"]), float(data["confidence"])
        except Exception:
            # Fallback: simple heuristic
            return False, 0.3
    
    def _ask_resource_conflict(self, a: SubTask, b: SubTask, original_task: str = "") -> Tuple[bool, List[str]]:
        """Check resource conflict between two tasks."""
        prompt = PromptManager.format_resource_conflict_prompt(original_task, a.description, b.description)
        
        try:
            data = self.llm_client.call_json(prompt, validator=DAGValidator.validate_resource_json)
            return bool(data["resource_conflict"]), list(data.get("shared_resources", []))
        except Exception:
            return False, []
    
    def _classify_modes(self, subtasks: List[SubTask], main_task: str):
        """Classify subtasks as WIDE or DEEP."""
        for st in subtasks:
            prompt = PromptManager.format_mode_classification_prompt(st.description, main_task)
            
            try:
                result = self.llm_client.call_json(prompt)
                st.mode = result.get("primary_mode", "DEEP")
            except Exception:
                # Fallback: heuristic classification
                desc_lower = st.description.lower()
                if any(keyword in desc_lower for keyword in ["search", "gather", "collect", "find", "explore"]):
                    st.mode = "WIDE"
                else:
                    st.mode = "DEEP"
    
    def _calculate_complexity(self, workflow: Workflow) -> ComplexityMetrics:
        """Calculate complexity metrics."""
        subtasks_info = [{"id": st.id, "desc": st.description, "mode": st.mode} for st in workflow.subtasks]
        prompt = PromptManager.format_complexity_analysis_prompt(
            workflow.original_prompt, 
            json.dumps(subtasks_info, ensure_ascii=False)
        )
        
        try:
            analysis = self.llm_client.call_json(prompt)
            
            # Calculate derived metrics
            parallel_efficiency = self._calculate_parallel_efficiency(workflow.dag)
            mode_heterogeneity = self._calculate_mode_heterogeneity(workflow.subtasks)
            
            base_score = (
                analysis.get("domain_complexity", 5) +
                analysis.get("coordination_complexity", 5) +
                analysis.get("computational_complexity", 5)
            ) / 3.0
            
            return ComplexityMetrics(
                base_score=base_score,
                uncertainty_score=analysis.get("overall_uncertainty", 0.5),
                domain_complexity=analysis.get("domain_complexity", 5),
                coordination_complexity=analysis.get("coordination_complexity", 5),
                computational_complexity=analysis.get("computational_complexity", 5),
                temporal_uncertainty=analysis.get("temporal_uncertainty", 5),
                resource_uncertainty=analysis.get("resource_uncertainty", 5),
                outcome_uncertainty=analysis.get("outcome_uncertainty", 5),
                critical_path_factor=self._calculate_critical_path_factor(workflow.dag),
                parallel_efficiency=parallel_efficiency,
                mode_heterogeneity=mode_heterogeneity,
                resource_conflict_factor=self._calculate_resource_conflict_factor(workflow.dag),
                requires_replanning=analysis.get("requires_replanning", False),
                risk_factors=analysis.get("risk_factors", []),
                mitigation_strategies=analysis.get("mitigation_strategies", [])
            )
            
        except Exception as e:
            print(f"Complexity analysis failed, using fallback: {str(e)}")
            return self._fallback_complexity_metrics(workflow)
    
    def _validate_quality(self, workflow: Workflow) -> WorkflowQualityMetrics:
        """Validate workflow quality."""
        # Use LLM validation
        llm_quality = self.validator.validate_quality_with_llm(
            workflow.original_prompt, 
            workflow.subtasks, 
            list(workflow.dag.edges())
        )
        
        # Combine with programmatic validation
        structural_issues = self.validator.validate_workflow_structure({
            'subtasks': workflow.subtasks,
            'dag': workflow.dag
        })
        
        # Calculate overall quality
        weights = [0.3, 0.25, 0.25, 0.2]  # completeness, coherence, efficiency, feasibility
        scores = [
            llm_quality["completeness_score"],
            llm_quality["coherence_score"], 
            llm_quality["efficiency_score"],
            llm_quality["feasibility_score"]
        ]
        overall_quality = sum(w * s for w, s in zip(weights, scores))
        
        return WorkflowQualityMetrics(
            completeness_score=llm_quality["completeness_score"],
            coherence_score=llm_quality["coherence_score"],
            efficiency_score=llm_quality["efficiency_score"],
            feasibility_score=llm_quality["feasibility_score"],
            overall_quality=overall_quality,
            validation_errors=llm_quality["validation_errors"] + structural_issues["structural_errors"],
            warnings=llm_quality["warnings"] + structural_issues["warnings"],
            suggestions=llm_quality["suggestions"] + structural_issues["suggestions"]
        )
    
    def _estimate_execution(self, workflow: Workflow) -> ExecutionEstimate:
        """Estimate execution metrics."""
        # Simple estimation based on task count and structure
        n_tasks = len(workflow.subtasks)
        wide_tasks = [st for st in workflow.subtasks if st.mode == "WIDE"]
        deep_tasks = [st for st in workflow.subtasks if st.mode == "DEEP"]
        
        # Time estimation
        wide_time = len(wide_tasks) * 1.5  # WIDE tasks take more time
        deep_time = len(deep_tasks) * 1.0
        total_sequential_time = wide_time + deep_time
        
        # Critical path time
        try:
            critical_path = nx.dag_longest_path(workflow.dag) if nx.is_directed_acyclic_graph(workflow.dag) else []
            critical_path_time = len(critical_path) * (total_sequential_time / n_tasks) if n_tasks > 0 else 1.0
        except:
            critical_path_time = total_sequential_time * 0.7
        
        parallel_savings = max(0, total_sequential_time - critical_path_time)
        
        # Resource requirements
        resource_requirements = {
            "cpu_cores": min(n_tasks, 4),
            "memory_gb": n_tasks * 2,
            "storage_gb": len(wide_tasks) * 10
        }
        
        cost_estimate = {
            "compute": n_tasks * 0.5,
            "storage": 0.1,
            "network": len(wide_tasks) * 0.2
        }
        
        return ExecutionEstimate(
            estimated_total_time=round(critical_path_time, 2),
            critical_path_time=round(critical_path_time, 2),
            parallel_time_savings=round(parallel_savings, 2),
            resource_requirements=resource_requirements,
            cost_estimate=cost_estimate,
            bottlenecks=["Estimation based on heuristics"]
        )
    
    def _diagnose_issues(self, workflow: Workflow, quality_metrics: WorkflowQualityMetrics, 
                        complexity_metrics: ComplexityMetrics) -> Dict[str, List[str]]:
        """Diagnose workflow issues."""
        issues = {
            "critical_issues": [],
            "performance_issues": [],
            "design_issues": [],
            "recommendations": []
        }
        
        # Critical issues
        if quality_metrics.overall_quality < 0.5:
            issues["critical_issues"].append("Overall workflow quality is poor")
        
        if complexity_metrics.requires_replanning:
            issues["critical_issues"].append("Complexity analysis suggests re-planning required")
        
        # Performance issues
        if complexity_metrics.parallel_efficiency < 0.3:
            issues["performance_issues"].append("Poor parallelization - workflow is too sequential")
        
        # Design issues
        if complexity_metrics.mode_heterogeneity > 0.8:
            issues["design_issues"].append("High mode heterogeneity - consider grouping similar tasks")
        
        # Recommendations
        if complexity_metrics.uncertainty_score > 0.7:
            issues["recommendations"].append("Add intermediate checkpoints due to high uncertainty")
        
        return issues
    
    def _generate_report(self, workflow: Workflow, complexity_metrics: ComplexityMetrics,
                        quality_metrics: WorkflowQualityMetrics, execution_estimate: ExecutionEstimate,
                        issues: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate comprehensive workflow report."""
        
        # Calculate parallel blocks
        parallel_blocks = {}
        for node in workflow.dag.nodes():
            level = workflow.dag.nodes[node].get('parallel_level', 0)
            if level not in parallel_blocks:
                parallel_blocks[level] = []
            parallel_blocks[level].append(node)
        
        return {
            "task_id": workflow.task_id,
            "complexity_score": complexity_metrics.base_score,
            "uncertainty_score": complexity_metrics.uncertainty_score,
            "quality_score": quality_metrics.overall_quality,
            "estimated_time_hours": execution_estimate.estimated_total_time,
            "parallel_time_savings": execution_estimate.parallel_time_savings,
            
            # Basic info
            "subtasks": [{"id": s.id, "desc": s.description, "mode": s.mode} for s in workflow.subtasks],
            "dependencies": list(workflow.dag.edges()),
            "parallel_blocks": parallel_blocks,
            "critical_path": nx.dag_longest_path(workflow.dag) if workflow.dag.nodes() else [],
            
            # Detailed metrics
            "complexity_metrics": {
                "base_score": complexity_metrics.base_score,
                "uncertainty_score": complexity_metrics.uncertainty_score,
                "parallel_efficiency": complexity_metrics.parallel_efficiency,
                "requires_replanning": complexity_metrics.requires_replanning,
                "risk_factors": complexity_metrics.risk_factors,
                "mitigation_strategies": complexity_metrics.mitigation_strategies
            },
            
            "quality_metrics": {
                "completeness": quality_metrics.completeness_score,
                "coherence": quality_metrics.coherence_score,
                "efficiency": quality_metrics.efficiency_score,
                "feasibility": quality_metrics.feasibility_score,
                "overall_quality": quality_metrics.overall_quality,
                "validation_errors": quality_metrics.validation_errors,
                "warnings": quality_metrics.warnings,
                "suggestions": quality_metrics.suggestions
            },
            
            "execution_metrics": {
                "estimated_total_time": execution_estimate.estimated_total_time,
                "critical_path_time": execution_estimate.critical_path_time,
                "parallel_savings": execution_estimate.parallel_time_savings,
                "resource_requirements": execution_estimate.resource_requirements,
                "cost_estimate": execution_estimate.cost_estimate,
                "bottlenecks": execution_estimate.bottlenecks
            },
            
            "issues": issues
        }
    
    # Helper methods
    def _calculate_parallel_efficiency(self, dag: nx.DiGraph) -> float:
        """Calculate parallel execution efficiency."""
        if dag.number_of_nodes() <= 1:
            return 1.0
        
        try:
            # Calculate level-based parallelization
            levels = {}
            for node in nx.topological_sort(dag):
                preds = list(dag.predecessors(node))
                levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
            
            max_level = max(levels.values()) if levels else 0
            return max(0.0, 1.0 - (max_level + 1) / dag.number_of_nodes())
        except:
            return 0.5
    
    def _calculate_mode_heterogeneity(self, subtasks: List[SubTask]) -> float:
        """Calculate mode heterogeneity."""
        if not subtasks:
            return 0.0
        
        wide_count = len([st for st in subtasks if st.mode == "WIDE"])
        total_count = len(subtasks)
        wide_ratio = wide_count / total_count
        
        # Maximum heterogeneity at 50/50 split
        return 2 * wide_ratio * (1 - wide_ratio)
    
    def _calculate_critical_path_factor(self, dag: nx.DiGraph) -> float:
        """Calculate critical path factor."""
        if dag.number_of_nodes() <= 1:
            return 1.0
        
        try:
            critical_path_length = len(nx.dag_longest_path(dag))
            return critical_path_length / dag.number_of_nodes()
        except:
            return 0.5
    
    def _calculate_resource_conflict_factor(self, dag: nx.DiGraph) -> float:
        """Calculate resource conflict factor."""
        if dag.number_of_edges() == 0:
            return 0.0
        
        conflict_edges = sum(1 for _, _, data in dag.edges(data=True) 
                           if data.get('has_resource_conflict', False))
        return conflict_edges / dag.number_of_edges()
    
    def _fallback_complexity_metrics(self, workflow: Workflow) -> ComplexityMetrics:
        """Generate fallback complexity metrics."""
        n_tasks = len(workflow.subtasks)
        base_score = min(20.0, n_tasks * 1.2 + 5.0)
        
        return ComplexityMetrics(
            base_score=base_score,
            uncertainty_score=0.5,
            domain_complexity=min(10, n_tasks + 3),
            coordination_complexity=min(10, workflow.dag.number_of_edges() + 2),
            computational_complexity=min(10, n_tasks + 2),
            temporal_uncertainty=5,
            resource_uncertainty=5,
            outcome_uncertainty=5,
            critical_path_factor=self._calculate_critical_path_factor(workflow.dag),
            parallel_efficiency=self._calculate_parallel_efficiency(workflow.dag),
            mode_heterogeneity=self._calculate_mode_heterogeneity(workflow.subtasks),
            resource_conflict_factor=self._calculate_resource_conflict_factor(workflow.dag),
            requires_replanning=base_score > 15.0,
            risk_factors=["Fallback analysis used"],
            mitigation_strategies=["Manual review recommended"]
        )
