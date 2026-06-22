"""
graph/reactflow_adapter.py

Converts a KnowledgeGraph into a ReactFlow-compatible payload.
This is the ONLY place in the Python codebase that knows about ReactFlow.
graph_schema.py and graph_meta.py remain completely clean of frontend concerns.

The layout (x/y positions) is intentionally left as (0, 0) here.
ReactFlow's elk/dagre layout engine on the frontend computes real positions.
"""

from graph.graph_schema import KnowledgeGraph, NodeType


# Map backend NodeType → ReactFlow custom node type string
# Must match the nodeTypes dict registered in your ReactFlow <ReactFlow> component.
NODE_TYPE_MAP: dict[NodeType, str] = {
    NodeType.PIPELINE:  "pipelineNode",
    NodeType.STAGE:     "stageNode",
    NodeType.ENTITY:    "entityNode",
    NodeType.OPERATION: "operationNode",
    NodeType.MODEL:     "modelNode",
    NodeType.METRIC:    "metricNode",
    NodeType.ATTRIBUTE: "attributeNode",
}


class ReactFlowAdapter:
    """
    Converts a KnowledgeGraph to the JSON shape ReactFlow expects.

    Output format:
        {
            "nodes": [ { id, type, data: { label, nodeType, ...props }, position } ],
            "edges": [ { id, source, target, label, type } ]
        }
    """

    def convert(self, graph: KnowledgeGraph) -> dict:
        rf_nodes = [
            {
                "id":       node.id,
                "type":     NODE_TYPE_MAP.get(node.node_type, "default"),
                "position": {"x": 0, "y": 0},   # layout engine sets this
                "data": {
                    "label":    node.label,
                    "nodeType": node.node_type.value,
                    **node.properties,
                },
            }
            for node in graph.nodes.values()
        ]

        rf_edges = [
            {
                "id":     f"{e.source_id}→{e.target_id}",
                "source": e.source_id,
                "target": e.target_id,
                "label":  e.label or e.edge_type.value,
                "type":   "smoothstep",
            }
            for e in graph.edges
        ]

        return {"nodes": rf_nodes, "edges": rf_edges}