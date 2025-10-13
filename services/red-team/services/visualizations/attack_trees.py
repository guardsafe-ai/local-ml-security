"""
Attack Tree Visualizer
Creates interactive attack trees showing attack paths, success rates, and dependencies.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
    import networkx as nx
    import numpy as np
except ImportError:
    # Fallback for environments without matplotlib/networkx
    class plt:
        @staticmethod
        def figure(*args, **kwargs): return None
        @staticmethod
        def subplot(*args, **kwargs): return None
        @staticmethod
        def show(*args, **kwargs): pass
        @staticmethod
        def savefig(*args, **kwargs): pass
        @staticmethod
        def close(*args, **kwargs): pass
    class patches:
        class FancyBboxPatch: pass
        class Circle: pass
        class Rectangle: pass
    class nx:
        class DiGraph: pass
    class np:
        @staticmethod
        def array(*args, **kwargs): return []
        @staticmethod
        def random(*args, **kwargs): return []

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of attack tree nodes"""
    ROOT = "root"
    AND = "and"
    OR = "or"
    LEAF = "leaf"
    ATTACK = "attack"
    VULNERABILITY = "vulnerability"
    COUNTERMEASURE = "countermeasure"

class NodeStatus(Enum):
    """Node status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class AttackNode:
    """Attack tree node"""
    node_id: str
    name: str
    description: str
    node_type: NodeType
    status: NodeStatus = NodeStatus.PENDING
    success_rate: float = 0.0
    difficulty: float = 0.0
    impact: float = 0.0
    cost: float = 0.0
    time_required: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[float, float] = (0.0, 0.0)

@dataclass
class AttackEdge:
    """Attack tree edge"""
    edge_id: str
    source: str
    target: str
    weight: float = 1.0
    label: str = ""
    style: str = "solid"
    color: str = "black"

class AttackTreeVisualizer:
    """Visualizes attack trees and attack paths"""
    
    def __init__(self):
        self.nodes: Dict[str, AttackNode] = {}
        self.edges: Dict[str, AttackEdge] = {}
        self.graph = nx.DiGraph()
        self.layout_positions: Dict[str, Tuple[float, float]] = {}
        
    def add_node(self, node: AttackNode):
        """Add node to attack tree"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **node.metadata)
        
    def add_edge(self, edge: AttackEdge):
        """Add edge to attack tree"""
        self.edges[edge.edge_id] = edge
        self.graph.add_edge(edge.source, edge.target, weight=edge.weight)
        
    def remove_node(self, node_id: str):
        """Remove node from attack tree"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.graph.remove_node(node_id)
            
            # Remove associated edges
            edges_to_remove = [
                edge_id for edge_id, edge in self.edges.items()
                if edge.source == node_id or edge.target == node_id
            ]
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
                self.graph.remove_edge(self.edges[edge_id].source, self.edges[edge_id].target)
    
    def create_attack_tree(self, attack_scenarios: List[Dict[str, Any]]) -> str:
        """Create attack tree from scenarios"""
        try:
            # Clear existing tree
            self.nodes.clear()
            self.edges.clear()
            self.graph.clear()
            
            # Create root node
            root_node = AttackNode(
                node_id="root",
                name="Attack Goal",
                description="Primary attack objective",
                node_type=NodeType.ROOT,
                success_rate=1.0
            )
            self.add_node(root_node)
            
            # Process attack scenarios
            for scenario in attack_scenarios:
                self._process_attack_scenario(scenario, root_node.node_id)
            
            # Calculate layout
            self._calculate_layout()
            
            # Generate visualization
            return self._generate_visualization()
            
        except Exception as e:
            logger.error(f"Failed to create attack tree: {e}")
            return ""
    
    def _process_attack_scenario(self, scenario: Dict[str, Any], parent_id: str):
        """Process individual attack scenario"""
        try:
            attack_id = scenario.get("attack_id", str(uuid.uuid4()))
            
            # Create attack node
            attack_node = AttackNode(
                node_id=attack_id,
                name=scenario.get("name", "Unknown Attack"),
                description=scenario.get("description", ""),
                node_type=NodeType.ATTACK,
                success_rate=scenario.get("success_rate", 0.0),
                difficulty=scenario.get("difficulty", 0.0),
                impact=scenario.get("impact", 0.0),
                cost=scenario.get("cost", 0.0),
                time_required=scenario.get("time_required", 0.0),
                parent=parent_id,
                metadata=scenario.get("metadata", {})
            )
            self.add_node(attack_node)
            
            # Create edge from parent
            edge = AttackEdge(
                edge_id=f"{parent_id}_{attack_id}",
                source=parent_id,
                target=attack_id,
                weight=attack_node.success_rate
            )
            self.add_edge(edge)
            
            # Process sub-attacks
            sub_attacks = scenario.get("sub_attacks", [])
            for sub_attack in sub_attacks:
                self._process_attack_scenario(sub_attack, attack_id)
            
            # Process vulnerabilities
            vulnerabilities = scenario.get("vulnerabilities", [])
            for vuln in vulnerabilities:
                self._process_vulnerability(vuln, attack_id)
            
            # Process countermeasures
            countermeasures = scenario.get("countermeasures", [])
            for countermeasure in countermeasures:
                self._process_countermeasure(countermeasure, attack_id)
                
        except Exception as e:
            logger.error(f"Failed to process attack scenario: {e}")
    
    def _process_vulnerability(self, vuln: Dict[str, Any], parent_id: str):
        """Process vulnerability node"""
        try:
            vuln_id = f"vuln_{str(uuid.uuid4())}"
            
            vuln_node = AttackNode(
                node_id=vuln_id,
                name=vuln.get("name", "Unknown Vulnerability"),
                description=vuln.get("description", ""),
                node_type=NodeType.VULNERABILITY,
                success_rate=vuln.get("exploitability", 0.0),
                difficulty=vuln.get("difficulty", 0.0),
                impact=vuln.get("impact", 0.0),
                parent=parent_id,
                metadata=vuln
            )
            self.add_node(vuln_node)
            
            # Create edge
            edge = AttackEdge(
                edge_id=f"{parent_id}_{vuln_id}",
                source=parent_id,
                target=vuln_id,
                weight=vuln_node.success_rate,
                color="red"
            )
            self.add_edge(edge)
            
        except Exception as e:
            logger.error(f"Failed to process vulnerability: {e}")
    
    def _process_countermeasure(self, countermeasure: Dict[str, Any], parent_id: str):
        """Process countermeasure node"""
        try:
            cm_id = f"cm_{str(uuid.uuid4())}"
            
            cm_node = AttackNode(
                node_id=cm_id,
                name=countermeasure.get("name", "Unknown Countermeasure"),
                description=countermeasure.get("description", ""),
                node_type=NodeType.COUNTERMEASURE,
                success_rate=countermeasure.get("effectiveness", 0.0),
                difficulty=countermeasure.get("implementation_difficulty", 0.0),
                impact=countermeasure.get("impact_reduction", 0.0),
                parent=parent_id,
                metadata=countermeasure
            )
            self.add_node(cm_node)
            
            # Create edge
            edge = AttackEdge(
                edge_id=f"{parent_id}_{cm_id}",
                source=parent_id,
                target=cm_id,
                weight=cm_node.success_rate,
                color="green"
            )
            self.add_edge(edge)
            
        except Exception as e:
            logger.error(f"Failed to process countermeasure: {e}")
    
    def _calculate_layout(self):
        """Calculate node positions for visualization"""
        try:
            if not self.nodes:
                return
            
            # Use hierarchical layout
            pos = nx.spring_layout(self.graph, k=3, iterations=50)
            
            # Adjust positions for better visualization
            levels = {}
            for node_id in self.graph.nodes():
                level = self._get_node_level(node_id)
                if level not in levels:
                    levels[level] = []
                levels[level].append(node_id)
            
            # Position nodes by level
            y_offset = 0
            for level in sorted(levels.keys()):
                level_nodes = levels[level]
                x_positions = np.linspace(-len(level_nodes)/2, len(level_nodes)/2, len(level_nodes))
                
                for i, node_id in enumerate(level_nodes):
                    pos[node_id] = (x_positions[i], -y_offset)
                
                y_offset += 2
            
            self.layout_positions = pos
            
        except Exception as e:
            logger.error(f"Failed to calculate layout: {e}")
    
    def _get_node_level(self, node_id: str) -> int:
        """Get node level in tree hierarchy"""
        level = 0
        current = node_id
        
        while current in self.nodes and self.nodes[current].parent:
            current = self.nodes[current].parent
            level += 1
        
        return level
    
    def _generate_visualization(self) -> str:
        """Generate attack tree visualization"""
        try:
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Draw edges
            for edge in self.edges.values():
                if edge.source in self.layout_positions and edge.target in self.layout_positions:
                    start_pos = self.layout_positions[edge.source]
                    end_pos = self.layout_positions[edge.target]
                    
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                           color=edge.color, linewidth=edge.weight * 3, alpha=0.7)
            
            # Draw nodes
            for node_id, node in self.nodes.items():
                if node_id in self.layout_positions:
                    pos = self.layout_positions[node_id]
                    self._draw_node(ax, node, pos)
            
            # Customize plot
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 2)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('Attack Tree Visualization', fontsize=16, fontweight='bold')
            
            # Add legend
            self._add_legend(ax)
            
            # Save to file
            filename = f"attack_tree_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to generate visualization: {e}")
            return ""
    
    def _draw_node(self, ax, node: AttackNode, pos: Tuple[float, float]):
        """Draw individual node"""
        try:
            x, y = pos
            
            # Choose color based on node type and status
            color = self._get_node_color(node)
            
            # Choose shape based on node type
            if node.node_type == NodeType.ROOT:
                shape = Circle((x, y), 0.5, color=color, alpha=0.8)
            elif node.node_type == NodeType.ATTACK:
                shape = Rectangle((x-0.4, y-0.3), 0.8, 0.6, color=color, alpha=0.8)
            elif node.node_type == NodeType.VULNERABILITY:
                shape = Circle((x, y), 0.3, color=color, alpha=0.8)
            elif node.node_type == NodeType.COUNTERMEASURE:
                shape = Rectangle((x-0.3, y-0.2), 0.6, 0.4, color=color, alpha=0.8)
            else:
                shape = Circle((x, y), 0.2, color=color, alpha=0.8)
            
            ax.add_patch(shape)
            
            # Add text
            ax.text(x, y, node.name[:10], ha='center', va='center', 
                   fontsize=8, fontweight='bold')
            
            # Add success rate
            if node.success_rate > 0:
                ax.text(x, y-0.4, f"{node.success_rate:.2f}", ha='center', va='center',
                       fontsize=6, color='white')
            
        except Exception as e:
            logger.error(f"Failed to draw node: {e}")
    
    def _get_node_color(self, node: AttackNode) -> str:
        """Get node color based on type and status"""
        if node.node_type == NodeType.ROOT:
            return "purple"
        elif node.node_type == NodeType.ATTACK:
            if node.status == NodeStatus.SUCCESS:
                return "red"
            elif node.status == NodeStatus.FAILED:
                return "gray"
            else:
                return "orange"
        elif node.node_type == NodeType.VULNERABILITY:
            return "red"
        elif node.node_type == NodeType.COUNTERMEASURE:
            return "green"
        else:
            return "blue"
    
    def _add_legend(self, ax):
        """Add legend to visualization"""
        try:
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor='purple', alpha=0.8, label='Root'),
                plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.8, label='Attack'),
                plt.Rectangle((0, 0), 1, 1, facecolor='orange', alpha=0.8, label='Attack (Pending)'),
                plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.8, label='Attack (Failed)'),
                plt.Circle((0, 0), 1, facecolor='red', alpha=0.8, label='Vulnerability'),
                plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.8, label='Countermeasure')
            ]
            
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
        except Exception as e:
            logger.error(f"Failed to add legend: {e}")
    
    def export_tree_data(self, format_type: str = "json") -> str:
        """Export attack tree data"""
        try:
            tree_data = {
                "nodes": [
                    {
                        "id": node.node_id,
                        "name": node.name,
                        "description": node.description,
                        "type": node.node_type.value,
                        "status": node.status.value,
                        "success_rate": node.success_rate,
                        "difficulty": node.difficulty,
                        "impact": node.impact,
                        "cost": node.cost,
                        "time_required": node.time_required,
                        "position": node.position,
                        "metadata": node.metadata
                    }
                    for node in self.nodes.values()
                ],
                "edges": [
                    {
                        "id": edge.edge_id,
                        "source": edge.source,
                        "target": edge.target,
                        "weight": edge.weight,
                        "label": edge.label,
                        "style": edge.style,
                        "color": edge.color
                    }
                    for edge in self.edges.values()
                ],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_nodes": len(self.nodes),
                    "total_edges": len(self.edges)
                }
            }
            
            if format_type == "json":
                return json.dumps(tree_data, indent=2)
            else:
                return str(tree_data)
                
        except Exception as e:
            logger.error(f"Failed to export tree data: {e}")
            return "{}"
    
    def calculate_attack_paths(self) -> List[List[str]]:
        """Calculate all possible attack paths"""
        try:
            paths = []
            
            # Find root nodes
            root_nodes = [
                node_id for node_id, node in self.nodes.items()
                if node.node_type == NodeType.ROOT
            ]
            
            for root in root_nodes:
                self._find_paths_from_node(root, [root], paths)
            
            return paths
            
        except Exception as e:
            logger.error(f"Failed to calculate attack paths: {e}")
            return []
    
    def _find_paths_from_node(self, node_id: str, current_path: List[str], all_paths: List[List[str]]):
        """Recursively find paths from a node"""
        try:
            # Find children
            children = [
                edge.target for edge in self.edges.values()
                if edge.source == node_id
            ]
            
            if not children:
                # Leaf node - add path
                all_paths.append(current_path.copy())
            else:
                # Continue with children
                for child in children:
                    if child not in current_path:  # Avoid cycles
                        current_path.append(child)
                        self._find_paths_from_node(child, current_path, all_paths)
                        current_path.pop()
                        
        except Exception as e:
            logger.error(f"Failed to find paths from node: {e}")
    
    def calculate_risk_scores(self) -> Dict[str, float]:
        """Calculate risk scores for each node"""
        try:
            risk_scores = {}
            
            for node_id, node in self.nodes.items():
                # Calculate risk as combination of impact, likelihood, and difficulty
                likelihood = node.success_rate
                impact = node.impact
                difficulty_factor = 1.0 - node.difficulty  # Lower difficulty = higher risk
                
                risk_score = likelihood * impact * difficulty_factor
                risk_scores[node_id] = risk_score
            
            return risk_scores
            
        except Exception as e:
            logger.error(f"Failed to calculate risk scores: {e}")
            return {}
    
    def get_critical_paths(self, threshold: float = 0.7) -> List[List[str]]:
        """Get critical attack paths above threshold"""
        try:
            risk_scores = self.calculate_risk_scores()
            all_paths = self.calculate_attack_paths()
            
            critical_paths = []
            for path in all_paths:
                path_risk = sum(risk_scores.get(node_id, 0) for node_id in path)
                if path_risk >= threshold:
                    critical_paths.append(path)
            
            return critical_paths
            
        except Exception as e:
            logger.error(f"Failed to get critical paths: {e}")
            return []
    
    def update_node_status(self, node_id: str, status: NodeStatus):
        """Update node status"""
        if node_id in self.nodes:
            self.nodes[node_id].status = status
            logger.info(f"Updated node {node_id} status to {status.value}")
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get attack tree statistics"""
        try:
            stats = {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": {},
                "status_counts": {},
                "avg_success_rate": 0.0,
                "avg_difficulty": 0.0,
                "avg_impact": 0.0,
                "total_paths": len(self.calculate_attack_paths()),
                "critical_paths": len(self.get_critical_paths())
            }
            
            # Count node types
            for node in self.nodes.values():
                node_type = node.node_type.value
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
                
                status = node.status.value
                stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1
            
            # Calculate averages
            if self.nodes:
                stats["avg_success_rate"] = sum(node.success_rate for node in self.nodes.values()) / len(self.nodes)
                stats["avg_difficulty"] = sum(node.difficulty for node in self.nodes.values()) / len(self.nodes)
                stats["avg_impact"] = sum(node.impact for node in self.nodes.values()) / len(self.nodes)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get tree statistics: {e}")
            return {}
