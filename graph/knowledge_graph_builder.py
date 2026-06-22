"""
graph/knowledge_graph_builder.py

Builds a KnowledgeGraph by reading ONLY the graph_meta key from each agent output.
Contains zero agent-specific logic. No imports from any agent module.
No conditionals about preprocessing, EDA, training, or any other stage.

Adding a new agent to the pipeline: implement build_graph_meta() in the agent,
store the result under state["agent_outputs"]["<name>"]["graph_meta"].
This file requires no changes.
"""

from graph.graph_schema import (
    KnowledgeGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)
from graph.graph_meta import GraphMeta, GraphNodeSpec, GraphEdgeSpec


class KnowledgeGraphBuilder:
    """
    Reads agent_outputs from PipelineState and builds a KnowledgeGraph.

    Contract:
        - Iterates over every key in agent_outputs.
        - Reads only agent_outputs[name]["graph_meta"].
        - Skips any agent that has no graph_meta (None or missing key).
        - Never reads any other field from any agent output.
    """

    def build(self, agent_outputs: dict) -> KnowledgeGraph:
        graph = KnowledgeGraph()

        # Optionally add a root pipeline node so every stage has somewhere to attach
        graph.add_node(GraphNode(
            id="pipeline",
            label="D.T.D Pipeline",
            node_type=NodeType.PIPELINE,
        ))

        for agent_name, output in agent_outputs.items():
            if not isinstance(output, dict):
                continue

            meta: GraphMeta | None = output.get("graph_meta")
            if meta is None:
                continue

            # 1. Register all nodes first (edges validated against existing nodes)
            for node_spec in meta.nodes:
                graph.add_node(self._node_from_spec(node_spec))

            # 2. Add explicit edges declared by the agent
            for edge_spec in meta.edges:
                graph.add_edge(self._edge_from_spec(edge_spec))

            # 3. Auto-generate CONTAINS edges from parent_id shortcuts
            for node_spec in meta.nodes:
                if node_spec.parent_id:
                    graph.add_edge(GraphEdge(
                        source_id=node_spec.parent_id,
                        target_id=node_spec.id,
                        edge_type=EdgeType.CONTAINS,
                    ))

        return graph

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _node_from_spec(spec: GraphNodeSpec) -> GraphNode:
        return GraphNode(
            id=spec.id,
            label=spec.label,
            node_type=spec.node_type,
            properties=spec.properties,
        )

    @staticmethod
    def _edge_from_spec(spec: GraphEdgeSpec) -> GraphEdge:
        return GraphEdge(
            source_id=spec.source_id,
            target_id=spec.target_id,
            edge_type=spec.edge_type,
            label=spec.label,
            properties=spec.properties,
        )