# graph/knowledge_graph_builder.py

from tools.graph_schema import (
    GraphNode,
    GraphEdge,
    node_to_dict,
    edge_to_dict,
)


def build_knowledge_graph(state):

    nodes = []
    edges = []

    flags = state.get("intent_flags", {})

    intent_node = GraphNode(
        id="intent",
        title="Intent Detection",
        status="completed",
        details={
            "target_column": state.get("target_column"),
            "task_type": state.get("task_type"),
        },
    )

    nodes.append(node_to_dict(intent_node))

    previous = "intent"

    agent_mapping = [
        ("run_eda", "eda", "EDA"),
        ("run_preprocessing", "preprocessing", "Preprocessing"),
        ("run_feature_engineering", "feature_engineering", "Feature Engineering"),
        ("run_model_selection", "model_selection", "Model Selection"),
        ("run_training", "training", "Training"),
        ("run_evaluation", "evaluation", "Evaluation"),
    ]

    for flag, node_id, title in agent_mapping:

        if flags.get(flag):

            node = GraphNode(
                id=node_id,
                title=title,
                status="waiting",
                details={}
            )

            nodes.append(node_to_dict(node))

            edges.append(
                edge_to_dict(
                    GraphEdge(
                        id=f"{previous}-{node_id}",
                        source=previous,
                        target=node_id,
                    )
                )
            )

            previous = node_id

    return {
        "nodes": nodes,
        "edges": edges,
    }