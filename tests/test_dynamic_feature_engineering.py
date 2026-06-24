"""
Feedback / HITL interrupt test — Feature Engineering Agent.

Runs the dynamic LangGraph pipeline through EDA and preprocessing, then pauses at
feature_engineering so YOU can type feedback interactively and confirm:
    1. The graph actually interrupts at the "feature_engineering" checkpoint.
    2. Typing feedback re-runs feature_engineering_node with your feedback_text
       included in the feature-generation prompt.
    3. Accepting moves the pipeline forward and records the feedback in
       state["feedback_history"].

EDA and preprocessing are auto-accepted so this script stays focused on Feature
Engineering only.

Usage:
    python tests/test_feedback_feature_engineering.py
    python tests/test_feedback_feature_engineering.py --data path/to.csv --target col --query "..."

At the feature_engineering checkpoint prompt, either:
    - press Enter (or type "accept")  -> approves and continues
    - type any other text             -> sent as feedback_text, feature_engineering_node re-runs

Example feedback:
    create ratio features only
    use interactions between Fare, Age, and family-size-like columns
    keep fewer weak features and prefer highly correlated ones
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

AGENT_NAME     = "feature_engineering"
DEFAULT_DATA   = str(PROJECT_ROOT / "assets/data/Classification Datasets/Titanic-Dataset.csv")
DEFAULT_TARGET = "Survived"
DEFAULT_QUERY  = "Run EDA, preprocessing, and feature engineering on this dataset. Do not run model selection, training, or evaluation."


def main():
    parser = argparse.ArgumentParser(description="Interactive feedback test for the Feature Engineering agent")
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

    feature_runs = []
    given_feedback = False
    iteration, max_steps = 0, 30

    while state.get("__interrupted__") and iteration < max_steps:
        paused_at = state["__paused_at__"]
        output = state.get("agent_outputs", {}).get(paused_at, {})

        print("\n" + "-" * 70)
        print(f"[PAUSED] checkpoint: '{paused_at}'")
        print("-" * 70)

        if paused_at == AGENT_NAME:
            feature_runs.append(output)
            print(json.dumps(output, indent=2, default=str)[:2500])
            selected = output.get("selected_features") or []
            if selected:
                print(f"\nselected_features: {selected}")
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
        print(f"\n[FEATURE ENGINEERING HISTORY] {len(feature_runs)} run(s):")
        for i, run in enumerate(feature_runs, 1):
            selected = run.get("selected_features") or []
            report = run.get("feature_report_path") or run.get("feature_summary_path")
            print(f"   {i}. status={run.get('status')} selected={selected} report={report}")

        history = [h for h in state.get("feedback_history", []) if h.get("agent") == AGENT_NAME]
        if history:
            print(f"\n[OK] feedback_history has {len(history)} entr(y/ies) for '{AGENT_NAME}':")
            for h in history:
                print(f"   - iteration {h.get('iteration')}: {h.get('feedback_text')}")
        elif given_feedback:
            print(f"\n[WARN] Feedback was sent but no entry found in feedback_history for '{AGENT_NAME}'.")
        else:
            print("\n[INFO] No feedback was given this run (you accepted immediately).")
    print("=" * 70)


if __name__ == "__main__":
    main()