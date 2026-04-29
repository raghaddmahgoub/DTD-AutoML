import json
from langchain_core.messages import SystemMessage, HumanMessage


class ControllerAgent:
    def __init__(self, logger, llm, registry):
        self.logger = logger
        self.llm = llm
        self.registry = registry

    def run(self, prompt: str):

        print("Plan Started")

        self.logger.info("\n" + "=" * 50)
        self.logger.info("LLM TOOL-CALLING AGENT MODE")
        self.logger.info("=" * 50)

        system_prompt = f"""
You are an AutoML agent that executes tasks step-by-step.

You MUST select ONE tool at a time and execute it.

Available tools:
{self.registry.list_tools()}

RULES:
- Return ONLY valid JSON
- Choose one tool per step
- Stop when task is complete

FORMAT:

{{
  "tool": "tool_name",
  "input": "input_data",
  "done": false
}}

FINAL STEP:

{{
  "tool": "none",
  "input": "",
  "done": true
}}
"""

        memory = prompt

        while True:

            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=memory)
            ])

            raw = response.content.strip()

            # remove ```json and ```
            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(raw)
            except Exception:
                self.logger.error("Failed to parse LLM output")
                self.logger.info("Raw output:")
                self.logger.info(response.content)
                return []

            tool_name = data.get("tool")
            tool_input = data.get("input")
            done = data.get("done", False)

            if done:
                self.logger.info("\n[AGENT] Workflow completed successfully")
                break

            tool = self.registry.get(tool_name)

            if tool is None:
                self.logger.error(f"Unknown tool: {tool_name}")
                break

            self.logger.info(f"\n[AGENT] Executing tool: {tool_name}")

            result = tool(tool_input)

            self.logger.info(f"[RESULT] {result}")

            # feed result back into LLM (memory loop)
            memory = f"""
Previous tool result:
{result}

Continue the AutoML workflow.
Task: {prompt}
"""