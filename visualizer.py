# visualizer.py
import pathlib
from typing import Dict, List
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from models import Workflow, ComplexityMetrics, WorkflowQualityMetrics, ExecutionEstimate
from utils import save_markdown


# ------------------------- DAG visualization ----------------------------- #

def create_visualization(workflow: Workflow, complexity_metrics: ComplexityMetrics, 
                                quality_metrics: WorkflowQualityMetrics, execution_estimate: ExecutionEstimate,
                                outpath: pathlib.Path):
    if workflow.dag.number_of_nodes() == 0:
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Main DAG visualization
    ax_main = plt.subplot2grid((4, 3), (0, 0), colspan=2, rowspan=3)
    
    # Color mapping based on multiple factors
    node_colors = []
    node_sizes = []
    edge_colors = []
    edge_widths = []
    
    # Get critical path for visualization
    try:
        critical_path = nx.dag_longest_path(workflow.dag) if nx.is_directed_acyclic_graph(workflow.dag) else []
    except:
        critical_path = []
    
    for node in workflow.dag.nodes():
        st = workflow.dag.nodes[node]["obj"]
        
        # Check if this is a group node
        is_group = workflow.dag.nodes[node].get('is_group', False)
        is_on_critical_path = node in critical_path
        
        if is_group:
            # Group nodes are light gray
            base_color = "lightgray"
        else:
            # Base color by mode
            if st.mode == "WIDE":
                base_color = "skyblue"
            elif st.mode == "DEEP":
                base_color = "lightgreen"
            else:  # Handle None or unknown modes
                base_color = "lightcoral"
            
            # Modify color based on critical path
            if is_on_critical_path:
                if st.mode == "WIDE":
                    base_color = "dodgerblue"
                elif st.mode == "DEEP":
                    base_color = "green"
                else:  # Handle None or unknown modes
                    base_color = "darkred"
        
        node_colors.append(base_color)
        
        # Node size based on estimated complexity
        base_size = 2000
        if 'parallel_peers' in workflow.dag.nodes[node]:
            peer_count = len(workflow.dag.nodes[node]['parallel_peers'])
            size_multiplier = 1.0 + (peer_count * 0.1)
            base_size = int(base_size * size_multiplier)
        
        node_sizes.append(base_size)
    
    # Create critical path edges set for fast lookup
    critical_path_edges = set()
    if len(critical_path) > 1:
        for i in range(len(critical_path) - 1):
            critical_path_edges.add((critical_path[i], critical_path[i+1]))
    
    # Edge styling based on confidence, conflicts, and critical path
    for edge in workflow.dag.edges(data=True):
        u, v = edge[0], edge[1]
        confidence = edge[2].get('confidence', 0.7)
        has_conflict = edge[2].get('has_resource_conflict', False)
        is_critical_edge = (u, v) in critical_path_edges
        
        if is_critical_edge:
            # Critical path edges are thick and dark blue
            edge_colors.append('darkblue')
            edge_widths.append(4.0)
        elif has_conflict:
            edge_colors.append('red')
            edge_widths.append(3.0)
        elif confidence > 0.8:
            edge_colors.append('darkgreen')
            edge_widths.append(2.0)
        elif confidence < 0.6:
            edge_colors.append('orange')
            edge_widths.append(1.5)
        else:
            edge_colors.append('gray')
            edge_widths.append(1.0)
    
    # Layout with hierarchical positioning
    pos = nx.spring_layout(workflow.dag, seed=42, k=3, iterations=50)
    
    # Draw the DAG
    nx.draw_networkx_nodes(workflow.dag, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.8, ax=ax_main)
    
    nx.draw_networkx_edges(workflow.dag, pos, edge_color=edge_colors, width=edge_widths,
                          alpha=0.7, arrows=True, arrowsize=20, ax=ax_main)
    
    # Labels with task IDs
    labels = {n: n for n in workflow.dag.nodes()}
    nx.draw_networkx_labels(workflow.dag, pos, labels, font_size=10, font_weight="bold", ax=ax_main)
    
    # Legend
    legend_elements = [
        Patch(facecolor="skyblue", edgecolor="black", label="WIDE Search"),
        Patch(facecolor="lightgreen", edgecolor="black", label="DEEP Reasoning"),
        Patch(facecolor="lightcoral", edgecolor="black", label="Unknown Mode"),
        Patch(facecolor="lightgray", edgecolor="black", label="Group Node"),
        Patch(facecolor="dodgerblue", edgecolor="black", label="WIDE (Critical Path)"),
        Patch(facecolor="green", edgecolor="black", label="DEEP (Critical Path)"),
        plt.Line2D([0], [0], color='darkblue', lw=4, label='Critical Path'),
        plt.Line2D([0], [0], color='darkgreen', lw=2, label='High Confidence Dependency'),
        plt.Line2D([0], [0], color='orange', lw=1.5, label='Low Confidence Dependency'),
        plt.Line2D([0], [0], color='red', lw=3, label='Resource Conflict')
    ]
    ax_main.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1))
    ax_main.set_title(f"Workflow DAG – Task {workflow.task_id}", fontsize=14, fontweight='bold')
    ax_main.axis('off')
    
    # Complexity metrics dashboard
    ax_complexity = plt.subplot2grid((4, 3), (0, 2))
    # Calculate average quality score
    avg_quality = (quality_metrics.completeness_score + quality_metrics.coherence_score + 
                  quality_metrics.efficiency_score + quality_metrics.feasibility_score) / 4.0
    
    complexity_data = [
        complexity_metrics.base_score / 20.0,  # Normalize to 0-1
        complexity_metrics.uncertainty_score,
        complexity_metrics.parallel_efficiency,
        avg_quality
    ]
    complexity_labels = ['Complexity', 'Uncertainty', 'Efficiency', 'Avg Quality']
    
    bars = ax_complexity.bar(complexity_labels, complexity_data, 
                           color=['red', 'orange', 'blue', 'green'])
    ax_complexity.set_ylim(0, 1)
    ax_complexity.set_title('Key Metrics', fontweight='bold')
    ax_complexity.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, complexity_data):
        height = bar.get_height()
        ax_complexity.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{value:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Execution timeline visualization
    ax_timeline = plt.subplot2grid((4, 3), (1, 2))
    
    # Create simplified timeline based on parallel levels
    levels = {}
    for node in workflow.dag.nodes():
        level = workflow.dag.nodes[node].get('parallel_level', 0)
        if level not in levels:
            levels[level] = []
        levels[level].append(node)
    
    max_level = max(levels.keys()) if levels else 0
    timeline_data = []
    timeline_labels = []
    
    for level in range(max_level + 1):
        level_tasks = levels.get(level, [])
        timeline_data.append(len(level_tasks))
        timeline_labels.append(f'Level {level}')
    
    ax_timeline.barh(range(len(timeline_data)), timeline_data, color='lightblue')
    ax_timeline.set_yticks(range(len(timeline_labels)))
    ax_timeline.set_yticklabels(timeline_labels)
    ax_timeline.set_xlabel('Parallel Tasks')
    ax_timeline.set_title('Execution Levels', fontweight='bold')
    
    # Resource requirements pie chart
    ax_resources = plt.subplot2grid((4, 3), (2, 2))
    
    resource_data = [
        execution_estimate.cost_estimate.get('compute', 1),
        execution_estimate.cost_estimate.get('storage', 0.1),
        execution_estimate.cost_estimate.get('network', 0.1)
    ]
    resource_labels = ['Compute', 'Storage', 'Network']
    
    ax_resources.pie(resource_data, labels=resource_labels, autopct='%1.1f%%', startangle=90)
    ax_resources.set_title('Resource Distribution', fontweight='bold')
    
    # Task details table
    ax_table = plt.subplot2grid((4, 3), (3, 0), colspan=3)
    ax_table.axis('off')
    
    # Create task summary table
    table_data = []
    group_nodes: set[str] = set()
    
    # First, collect all actual subtasks (non-group nodes)
    for st in workflow.subtasks:
        # Only include subtasks that actually exist in the DAG
        if st.id in workflow.dag.nodes:
            node_data = workflow.dag.nodes[st.id]
            is_group = node_data.get('is_group', False)
            
            if not is_group:
                parallel_level = node_data.get('parallel_level', 'N/A')
                on_critical = '✓' if node_data.get('on_critical_path', False) else ''
                peer_count = len(node_data.get('parallel_peers', []))
                
                table_data.append([
                    st.id,
                    st.mode or "Unknown",
                    f"Level {parallel_level}" if parallel_level != 'N/A' else 'N/A',
                    on_critical,
                    peer_count,
                    st.description[:50] + "..." if len(st.description) > 50 else st.description
                ])
            else:
                group_nodes.add(st.id)
    
    # Add any DAG nodes that aren't in the subtasks list (for hierarchical approaches)  
    for node_id in workflow.dag.nodes():
        if not any(st.id == node_id for st in workflow.subtasks):
            node_data = workflow.dag.nodes[node_id]
            is_group = node_data.get('is_group', False)
            st = node_data.get('obj')
            
            if st:
                if not is_group:
                    parallel_level = node_data.get('parallel_level', 'N/A')
                    on_critical = '✓' if node_data.get('on_critical_path', False) else ''
                    peer_count = len(node_data.get('parallel_peers', []))
                    
                    table_data.append([
                        st.id,
                        st.mode or "Unknown",
                        f"Level {parallel_level}" if parallel_level != 'N/A' else 'N/A',
                        on_critical,
                        peer_count,
                        st.description[:50] + "..." if len(st.description) > 50 else st.description
                    ])
                else:
                    group_nodes.add(st.id)
    
    # Add group node summaries at the end
    for group_id in group_nodes:
        # Find children of this group node
        children = list(workflow.dag.successors(group_id))
        actual_children = [c for c in children if not workflow.dag.nodes[c].get('is_group', False)]
        
        if actual_children:
            children_str = ", ".join(actual_children[:3])  # Show first 3 children
            if len(actual_children) > 3:
                children_str += f", +{len(actual_children) - 3} more"
            
            table_data.append([
                f"<{group_id}>",
                "GROUP",
                "N/A",
                "",
                len(actual_children),
                f"Contains: {children_str}"
            ])
    
    table_headers = ['Task ID', 'Mode', 'Parallel Level', 'Critical', 'Peers', 'Description']
    
    table = ax_table.table(cellText=table_data, colLabels=table_headers,
                          cellLoc='left', loc='center',
                          colColours=['lightgray'] * len(table_headers))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def create_metrics_report(workflow: Workflow, complexity_metrics: ComplexityMetrics,
                         quality_metrics: WorkflowQualityMetrics, execution_estimate: ExecutionEstimate,
                         issues: Dict[str, List[str]], outpath: pathlib.Path):
    report_content = f"""
# Workflow Analysis Report
Task ID: {workflow.task_id}
Original Task: {workflow.original_prompt}

## Executive Summary
- **Average Quality Score**: {(quality_metrics.completeness_score + quality_metrics.coherence_score + quality_metrics.efficiency_score + quality_metrics.feasibility_score) / 4.0:.3f}/1.0
- **Complexity Score**: {complexity_metrics.base_score:.2f}
- **Uncertainty Level**: {complexity_metrics.uncertainty_score:.3f}
- **Estimated Execution Time**: {execution_estimate.estimated_total_time:.1f} hours
- **Parallel Time Savings**: {execution_estimate.parallel_time_savings:.1f} hours

## Task Breakdown
Total Sub-tasks: {len(workflow.subtasks)}
- WIDE tasks: {len([st for st in workflow.subtasks if st.mode == "WIDE"])}
- DEEP tasks: {len([st for st in workflow.subtasks if st.mode == "DEEP"])}
- Unknown mode tasks: {len([st for st in workflow.subtasks if st.mode not in ["WIDE", "DEEP"]])}

Dependencies: {workflow.dag.number_of_edges()}
Parallel Levels: {len(set(workflow.dag.nodes[n].get('parallel_level', 0) for n in workflow.dag.nodes()))}

## Quality Assessment
- **Completeness**: {quality_metrics.completeness_score:.3f} - How well subtasks cover the main task
- **Coherence**: {quality_metrics.coherence_score:.3f} - Logical structure of dependencies
- **Efficiency**: {quality_metrics.efficiency_score:.3f} - Optimization for parallel execution
- **Feasibility**: {quality_metrics.feasibility_score:.3f} - Realistic achievability

## Complexity Analysis
- **Base Complexity**: {complexity_metrics.base_score:.2f}
- **Domain Complexity**: {complexity_metrics.domain_complexity}/10
- **Coordination Complexity**: {complexity_metrics.coordination_complexity}/10
- **Computational Complexity**: {complexity_metrics.computational_complexity}/10

### Uncertainty Factors
- **Temporal Uncertainty**: {complexity_metrics.temporal_uncertainty}/10
- **Resource Uncertainty**: {complexity_metrics.resource_uncertainty}/10
- **Outcome Uncertainty**: {complexity_metrics.outcome_uncertainty}/10

### Parallel Processing Analysis
- **Critical Path Factor**: {complexity_metrics.critical_path_factor:.3f}
- **Parallel Efficiency**: {complexity_metrics.parallel_efficiency:.3f}
- **Mode Heterogeneity**: {complexity_metrics.mode_heterogeneity:.3f}
- **Resource Conflicts**: {complexity_metrics.resource_conflict_factor:.3f}

## Execution Estimates
- **Sequential Time**: {execution_estimate.critical_path_time / complexity_metrics.parallel_efficiency if complexity_metrics.parallel_efficiency > 0 else 0:.1f} hours
- **Parallel Time**: {execution_estimate.estimated_total_time:.1f} hours
- **Time Savings**: {execution_estimate.parallel_time_savings:.1f} hours ({100 * execution_estimate.parallel_time_savings / max(execution_estimate.estimated_total_time + execution_estimate.parallel_time_savings, 1):.1f}% reduction)

### Resource Requirements
- **CPU Cores**: {execution_estimate.resource_requirements.get('cpu_cores', 1)}
- **Memory**: {execution_estimate.resource_requirements.get('memory_gb', 0):.1f} GB
- **Storage**: {execution_estimate.resource_requirements.get('storage_gb', 0):.1f} GB

## Issues and Recommendations

### Critical Issues
{chr(10).join(f"- {issue}" for issue in issues.get('critical_issues', []))}

### Performance Issues  
{chr(10).join(f"- {issue}" for issue in issues.get('performance_issues', []))}

### Design Issues
{chr(10).join(f"- {issue}" for issue in issues.get('design_issues', []))}

### Recommendations
{chr(10).join(f"- {rec}" for rec in issues.get('recommendations', []))}

## Risk Factors
{chr(10).join(f"- {risk}" for risk in complexity_metrics.risk_factors)}

## Mitigation Strategies
{chr(10).join(f"- {strategy}" for strategy in complexity_metrics.mitigation_strategies)}

## Re-planning Recommendation
{"⚠️ RE-PLANNING RECOMMENDED" if complexity_metrics.requires_replanning else "✅ Current plan appears adequate"}

---
Generated by Workflow Design System
"""
    
    save_markdown(report_content, outpath.with_suffix('.md'))
