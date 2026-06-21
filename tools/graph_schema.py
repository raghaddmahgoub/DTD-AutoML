# graph/graph_schema.py

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    id: str
    title: str
    type: str = "agent"
    status: str = "waiting"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    id: str
    source: str
    target: str


def node_to_dict(node: GraphNode):
    return {
        "id": node.id,
        "title": node.title,
        "type": node.type,
        "status": node.status,
        "details": node.details,
    }


def edge_to_dict(edge: GraphEdge):
    return {
        "id": edge.id,
        "source": edge.source,
        "target": edge.target,
    }