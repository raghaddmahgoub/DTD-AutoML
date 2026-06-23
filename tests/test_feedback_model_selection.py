"""
Feedback / HITL interrupt test — Model Selection Agent.

Runs the dynamic LangGraph pipeline up to the model_selection checkpoint,
pauses there, and lets YOU type feedback interactively so you can confirm:
    1. The graph actually interrupts at the "model_selection" checkpoint.
    2. Typing feedback re-runs model_selection_node with your feedback_text
       injected (e.g. ask it to prefer a different training approach/model
       and see training_plan / plan_preview change between the two runs).
    3. Accepting moves the pipeline forward and records the feedback in
       state["feedback_history"].

EDA + Preprocessing are enabled too (model selection needs the dataset
profile + preprocessed paths), but their checkpoints are auto-accepted so
this script stays focused on Model Selection.

Usage:
    python tests/test_feedback_model_selection.py
    python tests/test_feedback_model_selection.py --data path/to.csv --target col --query "..."

At the model_selection checkpoint prompt, either:
    - press Enter (or type "accept")  -> approves and continues
    - type any other text             -> sent as feedback_text, model_selection_node re-runs
"""
import argparse
import json
import os
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from agents.dynamic.controller_agent.controller_agent import ControllerAgent
from src.utils.logger import Logger

AGENT_NAME     = "model_selection"
DEFAULT_DATA   = str(PROJECT_ROOT / "assets/data/Datasets/Classification Datasets/Iris.csv")
DEFAULT_TARGET = "species"
DEFAULT_QUERY  = (
    "Run EDA, preprocessing, and model selection planning on this dataset. "
    "Do not train or evaluate yet."
)


def main():
    parser = argparse.ArgumentParser(description="Interactive feedback test for the Model Selection agent")
    parser.add_argument("--data",   default=DEFAULT_DATA)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--query",  default=DEFAULT_QUERY)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    Logger()

    if not Path(args.data).exists():
        print(f"[ERROR] Dataset not found: {args.data}")
        sys.exit(1)

    controller = ControllerAgent()
    run_id = f"test-feedback-{AGENT_NAME}-{os.urandom(3).hex()}"

    print("=" * 70)
    print(f"FEEDBACK TEST — {AGENT_NAME.upper()} AGENT")
    print("=" * 70)
    print(f"data   : {args.data}")
    print(f"target : {args.target}")
    print(f"query  : {args.query}")
    print(f"run_id : {run_id}")
    print("-" * 70)

    state = controller.run({
        "data_path":     args.data,
        "prompt":        args.query,
        "target_column": args.target,
        "run_id":        run_id,
    })

    given_feedback = False
    iteration, max_steps = 0, 30

    while state.get("__interrupted__") and iteration < max_steps:
        paused_at = state["__paused_at__"]
        output = state.get("agent_outputs", {}).get(paused_at, {})

        print("\n" + "-" * 70)
        print(f"[PAUSED] checkpoint: '{paused_at}'")
        print("-" * 70)

        if paused_at == AGENT_NAME:
            print(json.dumps(output, indent=2, default=str)[:2000])
            print("-" * 70)
            choice = input(
                f"Enter feedback for '{AGENT_NAME}' (Enter / 'accept' to approve): "
            ).strip()
            if choice and choice.lower() != "accept":
                decision, feedback_text = "feedback", choice
                given_feedback = True
                print(f"[SENDING FEEDBACK] '{feedback_text}'")
            else:
                decision, feedback_text = "accept", ""
                print("[ACCEPTED]")
        else:
            decision, feedback_text = "accept", ""
            print(f"[AUTO-ACCEPT] (not under test) — preview: {str(output)[:200]}")

        state = controller.resume(run_id=run_id, decision=decision, feedback_text=feedback_text)
        iteration += 1

    print("\n" + "=" * 70)
    if state.get("__error__"):
        print(f"[ERROR] Pipeline failed: {state['__error__']}")
    else:
        print("[DONE] Pipeline finished (or max steps reached).")
        history = [h for h in state.get("feedback_history", []) if h.get("agent") == AGENT_NAME]
        if history:
            print(f"[OK] feedback_history has {len(history)} entr(y/ies) for '{AGENT_NAME}':")
            for h in history:
                print(f"   - iteration {h.get('iteration')}: {h.get('feedback_text')}")
        elif given_feedback:
            print(f"[WARN] Feedback was sent but no entry found in feedback_history for '{AGENT_NAME}'.")
        else:
            print("[INFO] No feedback was given this run (you accepted immediately).")
    print("=" * 70)


if __name__ == "__main__":
    main()
