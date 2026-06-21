import json
from langchain_core.messages import SystemMessage, HumanMessage

from tools.pipeline_state import empty_state, ensure_state, merge_state


class ControllerAgent:
    def __init__(self, logger, llm, registry):
        self.logger = logger
        self.llm = llm
        self.registry = registry
        print("ControllerAgent initialized with tools:", self.registry.list_tools())

    def run(self, inputs:dict):
        data_path = inputs.get("data_path")
        target_column = inputs.get("target_column")
        prompt = inputs.get("prompt")

        print("Plan Started")

        self.logger.info("\n" + "=" * 50)
        self.logger.info("LLM TOOL-CALLING AGENT MODE")
        self.logger.info("=" * 50)

        system_prompt = f"""
You are an AutoML controller that executes ONE tool per step.

Available tools:
{self.registry.list_tools_with_schema()}

PIPELINE ORDER (follow when possible):
1) preprocessing_execution  (required — produces train/test splits)
2) feature_engineering_execution  (runs inside PreprocessingAgent graph; adds engineered features)
3) plan_training  (builds plan + user approval; does NOT train)
4) exactly ONE training tool from plan:
   - train_simple         → default sklearn hyperparameters
   - train_simple_optuna  → sklearn + Optuna HPO
   - train_autogluon      → AutoGluon AutoML
   Training tools require preprocessed splits in pipeline_state (no raw-data preprocessing).
5) evaluate

RULES:
- Return ONLY valid JSON
- Choose one tool per step
- Set "task" to a concrete instruction for the tool you call (passed to its LLM like dynamic EDA)
- When choosing plan_training for model training, include training-critical details from the user prompt in "task":
  target, metric, speed/quality, preferred or excluded models, AutoGluon/Optuna requests,
  interpretability, hardware, and deployment constraints.
- Do not call train_* before preprocessing and plan_training are complete
- After plan_training, call the train tool named in result.train_tool
- Stop when evaluation is done

FORMAT:
{{
  "tool": "tool_name",
  "task": "task_description",
  "input": {{}},
  "done": false
}}

FINAL STEP:
{{
  "tool": "none",
  "task": "",
  "input": {{}},
  "done": true
}}
"""

        pipeline_state = empty_state(data_path, prompt)
        pipeline_state["target_column"] = target_column
        memory = f"Task: {prompt}\nPipeline state step: {pipeline_state.get('step')}"

        while True:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=memory),
            ])

            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                step = json.loads(raw)
            except Exception:
                self.logger.error("Failed to parse LLM output")
                self.logger.info("Raw output:")
                self.logger.info(response.content)
                return pipeline_state

            tool_name = step.get("tool")
            tool_input = step.get("input", {})
            task = step.get("task")
            done = step.get("done", False)

            if done:
                self.logger.info("\n[AGENT] Workflow completed successfully")
                pipeline_state["status"] = "completed"
                break

            tool = self.registry.get(tool_name)
            if tool is None:
                self.logger.error(f"Unknown tool: {tool_name}")
                break

            self.logger.info(f"\n[AGENT] Executing tool: {tool_name}")

            if task:
                pipeline_state = merge_state(pipeline_state, {"controller_task": task})

            result, pipeline_state = tool.invoke({
                "task": task,
                "tool_input": tool_input,
                "target_column": pipeline_state.get("target_column", target_column),
                "prompt": prompt,
                "data_path": pipeline_state.get("data_path", data_path),
                "llm": self.llm,
                "state": pipeline_state,
            })

            # Backward compatibility: if a tool still returns only data_path string
            if isinstance(pipeline_state, str):
                pipeline_state = ensure_state(None, pipeline_state, prompt)

            self.logger.info(f"[RESULT] {result}")

            memory = f"""
User prompt: {prompt}

Last tool: {tool_name}
Last controller task: {task or '(none)'}
Last result: {json.dumps(result, default=str)[:2500]}
Pipeline step: {pipeline_state.get('step')}
Training plan approved: {(pipeline_state.get('training_plan') or {}).get('approved')}
Next train tool (if planned): {(pipeline_state.get('training_plan') or {}).get('train_tool')}

Choose the NEXT single tool. Include a specific "task" string for that tool.
"""

        return pipeline_state
