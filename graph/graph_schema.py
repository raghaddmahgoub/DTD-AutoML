"""
graph/graph_schema.py

Core graph primitives. Zero knowledge of agents, ReactFlow, or any pipeline stage.
These are the only data structures the KnowledgeGraph understands internally.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class NodeType(str, Enum):
    PIPELINE   = "pipeline"    # Top-level pipeline root
    STAGE      = "stage"       # A pipeline stage (EDA, Preprocessing, …)
    ENTITY     = "entity"      # A data entity (column, feature, …)
    OPERATION  = "operation"   # A transformation or action applied to an entity
    MODEL      = "model"       # A candidate or selected ML model
    METRIC     = "metric"      # An evaluation metric value
    ATTRIBUTE  = "attribute"   # A named statistic or property


class EdgeType(str, Enum):
    CONTAINS   = "contains"    # Parent → child structural relationship
    PRODUCES   = "produces"    # Stage → output entity
    APPLIES    = "applies"     # Entity → operation applied to it
    SELECTED   = "selected"    # Model selection decision
    FEEDS      = "feeds"       # Data flows from one stage to the next
    CORRELATES = "correlates"  # Statistical relationship between entities


# ─────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────

@dataclass
class GraphNode:
    id:         str
    label:      str
    node_type:  NodeType
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id:  str
    target_id:  str
    edge_type:  EdgeType
    label:      Optional[str] = None
    properties: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Graph container
# ─────────────────────────────────────────────

@dataclass
class KnowledgeGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge]      = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        """Add a node. Silently overwrites if id already exists."""
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """
        Add an edge. Skips silently if either endpoint node is missing
        (prevents dangling edges from partial pipeline runs).
        """
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return
        self.edges.append(edge)

    def to_dict(self) -> dict:
        """Serialisable snapshot — used for SSE payloads and caching."""
        return {
            "nodes": [
                {
                    "id":         n.id,
                    "label":      n.label,
                    "node_type":  n.node_type.value,
                    "properties": n.properties,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source_id":  e.source_id,
                    "target_id":  e.target_id,
                    "edge_type":  e.edge_type.value,
                    "label":      e.label,
                    "properties": e.properties,
                }
                for e in self.edges
            ],
        }