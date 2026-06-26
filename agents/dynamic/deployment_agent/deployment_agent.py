"""Deployment Agent — LangGraph orchestrator nodes and routing."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from state.pipeline_state import PipelineState
from tools.shared import get_llm
from tools.shared.prompt_builder import build_prompt_deployment
from graph.knowledge_graph import update_agent_progress

logger = logging.getLogger(__name__)


class DeploymentFiles(BaseModel):
    """Pydantic model for structured LLM code generation."""
    api_server_code: str = Field(description="FastAPI python server code (api_server.py) with Pydantic validation and /predict + /health endpoints.")
    dockerfile_code: str = Field(description="Production Dockerfile using python:3.11-slim, copying app, installing dependencies, and exposing port 8000.")
    requirements_txt: str = Field(description="requirements.txt containing fastapi, uvicorn, scikit-learn (or other required ML frameworks), pandas, numpy, etc.")


def _build_feedback_context(state: PipelineState, agent_name: str) -> str:
    """Pull feedback history specifically for this agent."""
    history = state.get("feedback_history", []) or []
    own = [h["feedback_text"] for h in history if h.get("agent") == agent_name]
    if own:
        return f"\n\nUser Feedback History for {agent_name}:\n" + "\n".join(f"- {f}" for f in own)
    return ""


class DeploymentAgent:
    """Coordinates code generation and packaging of models for serving."""

    def __init__(self, logger_obj: Any = None, llm: Any = None):
        self.logger = logger_obj or logger
        base_llm = llm or get_llm()
        self.llm = base_llm.with_structured_output(DeploymentFiles)

    def run(self, pipeline_state: PipelineState) -> dict:
        run_id = pipeline_state.get("run_id") or "run-default"
        logger.info("[DeploymentAgent] Starting deployment package generation for %s", run_id)

        sub_nodes = [
            {"name": "Initialization", "description": "Extracting feature schema and configuration.", "status": "pending"},
            {"name": "App Generation", "description": "Generating FastAPI serving application using LLM.", "status": "pending"},
            {"name": "Docker Generation", "description": "Generating Docker container config and dependencies.", "status": "pending"},
            {"name": "Save Package", "description": "Saving all deployment package files to output folder.", "status": "pending"},
        ]
        agent_output = {
            "status": "running",
            "sub_nodes": sub_nodes
        }

        # Initialize progress in DB
        update_agent_progress(run_id, "deployment", agent_output)

        def update_step(name: str, status: str, description: str = None):
            for node in sub_nodes:
                if node["name"] == name:
                    node["status"] = status
                    if description:
                        node["description"] = description
                    break
            update_agent_progress(run_id, "deployment", agent_output)

        # ── Step 1: Initialization ───────────────────────────────────────────
        update_step("Initialization", "running")

        trained_model_path = pipeline_state.get("trained_model_path")
        if not trained_model_path or not Path(trained_model_path).exists():
            error_msg = f"No trained model pickle found at {trained_model_path}. Cannot deploy."
            logger.error("[DeploymentAgent] %s", error_msg)
            update_step("Initialization", "failed", error_msg)
            for node in sub_nodes[1:]:
                node["status"] = "skipped"
            agent_output["status"] = "failed"
            agent_output["error"] = error_msg
            update_agent_progress(run_id, "deployment", agent_output)

            outputs = dict(pipeline_state.get("agent_outputs", {}))
            outputs["deployment"] = agent_output
            pipeline_state["agent_outputs"] = outputs
            pipeline_state["error"] = error_msg
            return pipeline_state

        # Extract feature schema from training splits to provide schema context
        feature_schema = {}
        X_train_path = pipeline_state.get("X_train_engineered_path") or pipeline_state.get("X_train_path")
        try:
            import pandas as pd
            if X_train_path and Path(X_train_path).exists():
                df_sample = pd.read_csv(X_train_path, nrows=5)
                feature_schema = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
            else:
                data_path = pipeline_state.get("data_path")
                if data_path and Path(data_path).exists():
                    df_sample = pd.read_csv(data_path, nrows=5)
                    target = pipeline_state.get("target_column")
                    if target and target in df_sample.columns:
                        df_sample = df_sample.drop(columns=[target])
                    feature_schema = {col: str(dtype) for col, dtype in df_sample.dtypes.items()}
        except Exception as e:
            logger.warning("[DeploymentAgent] Could not parse feature schema from files: %s", e)

        task_type = pipeline_state.get("task_type") or pipeline_state.get("problem_type") or "classification"
        model_metrics = pipeline_state.get("model_metrics") or {}

        update_step("Initialization", "completed", f"Extracted schema with {len(feature_schema)} features.")

        # ── Step 2: Code & Dockerfile Generation ─────────────────────────────────
        update_step("App Generation", "running")
        update_step("Docker Generation", "running")

        feedback = _build_feedback_context(pipeline_state, "deployment")
        prompts = build_prompt_deployment(
            trained_model_path=trained_model_path,
            task_type=task_type,
            feature_schema=feature_schema,
            model_metrics=model_metrics,
            feedback_context=feedback,
        )

        try:
            response: DeploymentFiles = self.llm.invoke([
                SystemMessage(content=prompts.system),
                HumanMessage(content=prompts.user),
            ])
            api_server_code = response.api_server_code
            dockerfile_code = response.dockerfile_code
            requirements_txt = response.requirements_txt

            update_step("App Generation", "completed", "Generated FastAPI python serving script (api_server.py).")
            update_step("Docker Generation", "completed", "Generated Dockerfile and requirements dependencies.")

        except Exception as exc:
            error_msg = f"LLM generation failed: {exc}"
            logger.error("[DeploymentAgent] %s", error_msg)
            update_step("App Generation", "failed", error_msg)
            update_step("Docker Generation", "skipped")
            update_step("Save Package", "skipped")
            agent_output["status"] = "failed"
            agent_output["error"] = error_msg
            update_agent_progress(run_id, "deployment", agent_output)

            outputs = dict(pipeline_state.get("agent_outputs", {}))
            outputs["deployment"] = agent_output
            pipeline_state["agent_outputs"] = outputs
            pipeline_state["error"] = error_msg
            return pipeline_state

        # ── Step 3: Save Package ─────────────────────────────────────────────
        update_step("Save Package", "running")
        try:
            deploy_dir = Path("output") / "deployment" / run_id
            deploy_dir.mkdir(parents=True, exist_ok=True)

            # Write code and configs to folder
            with open(deploy_dir / "api_server.py", "w", encoding="utf-8") as f:
                f.write(api_server_code)

            with open(deploy_dir / "Dockerfile", "w", encoding="utf-8") as f:
                f.write(dockerfile_code)

            with open(deploy_dir / "requirements.txt", "w", encoding="utf-8") as f:
                f.write(requirements_txt)

            # Copy model pkl file to same folder
            shutil.copy2(trained_model_path, deploy_dir / "model.pkl")
            import re
            version_match = re.search(r'(?:MODEL_)?VERSION\s*=\s*["\']([\w\.]+)["\']', api_server_code, re.IGNORECASE)
            version = version_match.group(1) if version_match else "1.0"

            pkl_matches = re.findall(r'f?["\'](model_[\w\.\-\{\}]+?\.pkl)["\']', api_server_code)
            for match in pkl_matches:
                resolved_name = re.sub(r'\{[\w_]+\}', version, match)
                try:
                    shutil.copy2(trained_model_path, deploy_dir / resolved_name)
                    logger.info("[DeploymentAgent] Copied model to matching name: %s", resolved_name)
                except Exception as e:
                    logger.warning("[DeploymentAgent] Could not copy to matching name %s: %s", resolved_name, e)

            # Set outputs in state
            pipeline_state["deployment_package_path"] = str(deploy_dir.resolve())
            pipeline_state["endpoint_url"] = "http://localhost:8000"

            update_step("Save Package", "completed", f"Saved full deployment bundle in: {deploy_dir}.")
            status_val = "success"

        except Exception as exc:
            error_msg = f"Failed to save deployment package files: {exc}"
            logger.error("[DeploymentAgent] %s", error_msg)
            update_step("Save Package", "failed", error_msg)
            status_val = "failed"
            pipeline_state["error"] = error_msg

        # Final UI report payload
        agent_output = {
            "status": status_val,
            "deployment_package_path": pipeline_state.get("deployment_package_path"),
            "endpoint_url": pipeline_state.get("endpoint_url"),
            "api_server_code": api_server_code if status_val == "success" else None,
            "dockerfile_code": dockerfile_code if status_val == "success" else None,
            "requirements_txt": requirements_txt if status_val == "success" else None,
            "sub_nodes": sub_nodes,
        }
        update_agent_progress(run_id, "deployment", agent_output)

        merged_outputs = dict(pipeline_state.get("agent_outputs", {}))
        merged_outputs["deployment"] = agent_output
        pipeline_state["agent_outputs"] = merged_outputs
        pipeline_state["step"] = "deployment_completed" if status_val == "success" else "deployment_failed"
        pipeline_state["status"] = "success" if status_val == "success" else "error"

        return pipeline_state


def deployment_node(state: PipelineState) -> dict:
    """LangGraph node execution wrapper."""
    agent = DeploymentAgent()
    return agent.run(state)


def route_after_deployment(state: PipelineState) -> str:
    """Dynamic conditional edge routing after deployment checkpoint."""
    # Terminal step in current pipeline graph
    return "pipeline_done"
