import json
from langchain_core.messages import SystemMessage, HumanMessage


class ControllerAgent:
    def __init__(self, logger, llm, registry):
        self.logger = logger
        self.llm = llm
        self.registry = registry

    def run(self,data_path: str, prompt: str):

        print("Plan Started")

        self.logger.info("\n" + "=" * 50)
        self.logger.info("LLM TOOL-CALLING AGENT MODE")
        self.logger.info("=" * 50)

        system_prompt = f"""
You are an AutoML agent that executes tasks step-by-step.

You MUST select ONE tool at a time and execute it.

Available tools:
{self.registry.list_tools_with_schema()}

RULES:
- Return ONLY valid JSON
- Choose one tool per step
- Stop when task is complete
- Be logical in ordering (understand → clean → features → train → evaluate)

FORMAT:

{{
  "tool": "tool_name",
  "task": "task_description",
  "input": "input_data_for_that_tool",
  "done": false
}}

FINAL STEP:

{{
  "tool": "none",
  "task": "",
  "input": "",
  "done": true
}}
"""

        memory = f"Task: {prompt}"
        data = data_path  # shared pipeline data

        while True:

            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=memory)
            ])

            raw = response.content.strip()

            # clean markdown
            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                step = json.loads(raw)
            except Exception:
                self.logger.error("Failed to parse LLM output")
                self.logger.info("Raw output:")
                self.logger.info(response.content)
                return []

            tool_name = step.get("tool")
            tool_input = step.get("input")
            task = step.get("task")
            done = step.get("done", False)

            if done:
                self.logger.info("\n[AGENT] Workflow completed successfully")
                break

            tool = self.registry.get(tool_name)

            if tool is None:
                self.logger.error(f"Unknown tool: {tool_name}")
                break

            self.logger.info(f"\n[AGENT] Executing tool: {tool_name}")

            # 🔥 KEY CHANGE: pass BOTH input + ORIGINAL PROMPT
            result , data_path = tool.invoke({"task": task, "tool_input": tool_input, "prompt": prompt, "data_path": data_path,"llm": self.llm})
            print("***********************")
            print(data_path)
            self.logger.info(f"[RESULT] {result}")

            # update shared data if tool returns something meaningful
            if result is not None:
                data = result

            # memory loop (gives context to next step)
            memory = f"""
Task: {prompt}

Last tool used: {tool_name}
Tool input: {tool_input}
Tool result: {result}

What is the NEXT best tool?
Remember: choose ONLY ONE tool.
"""