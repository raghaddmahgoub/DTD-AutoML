"""
graph/graph_meta.py

The contract every agent output can optionally expose.
Agents import from here — they never import graph_schema.py directly.
The KnowledgeGraphBuilder reads only this contract; it never reads agent domain fields.

Usage inside an agent:
    from graph.graph_meta import GraphMeta, GraphNodeSpec, GraphEdgeSpec
    from graph.graph_schema import NodeType, EdgeType

    def build_graph_meta(output: dict) -> GraphMeta:
        ...
        return GraphMeta(nodes=[...], edges=[...])

    # Then inside the agent's LangGraph node function:
    state["agent_outputs"]["my_agent"]["graph_meta"] = build_graph_meta(output)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from graph.graph_schema import NodeType, EdgeType


@dataclass
class GraphNodeSpec:
    """
    Declarative spec for one node.

    Agents fill these fields — they never instantiate GraphNode directly.
    The builder converts specs to real GraphNodes.

    parent_id:
        Optional shortcut. When set, the builder automatically creates a
        CONTAINS edge from parent_id → this node's id.
        Use instead of manually declaring a GraphEdgeSpec for simple hierarchy.
    """
    id:         str
    label:      str
    node_type:  NodeType
    properties: dict[str, Any]  = field(default_factory=dict)
    parent_id:  Optional[str]   = None


@dataclass
class GraphEdgeSpec:
    """
    Declarative spec for one edge.

    Use when you need explicit control over edge type and label.
    For simple parent→child hierarchy, prefer setting parent_id on GraphNodeSpec.
    """
    source_id:  str
    target_id:  str
    edge_type:  EdgeType
    label:      Optional[str]   = None
    properties: dict[str, Any]  = field(default_factory=dict)


@dataclass
class GraphMeta:
    """
    The graph metadata contract.

    This is the ONLY thing the KnowledgeGraphBuilder reads from agent_outputs.
    Every other field in the agent output dict is invisible to the builder.

    An agent that does not set graph_meta is simply ignored by the builder —
    no error, no crash, no required changes to the builder.
    """
    nodes: list[GraphNodeSpec] = field(default_factory=list)
    edges: list[GraphEdgeSpec] = field(default_factory=list)