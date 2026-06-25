"""
Feedback / HITL interrupt test — Model Selection + Training loop.

Training itself can't reliably turn free-text feedback ("use XGBoost
instead") into a concrete model change — train_simple/train_simple_optuna/
train_autogluon just execute whatever training_plan was already approved.
So feedback given on the TRAINING checkpoint is rerouted by graph_builder.py
back to model_selection_agent, which already has a reliable LLM step
(tools/training/model_selection.py) to turn that feedback into a new
training_plan (different model / approach). That replanned graph then flows
forward through model_selection_checkpoint -> training_agent again, so the
model actually gets retrained differently.

This script lets you exercise BOTH feedback points in one run:
    1. model_selection checkpoint — feedback here re-plans directly
       (e.g. "use Simple+Optuna instead of plain RandomForest").
    2. training checkpoint — feedback here reroutes to model_selection_agent
       to re-plan based on what you saw in the trained result
       (e.g. "the accuracy is low, try XGBoost"), then re-trains.

Every OTHER checkpoint (EDA, preprocessing, evaluation) is auto-accepted so
this script stays focused on the model_selection <-> training loop.

Usage:
    python tests/test_feedback_training.py
    python tests/test_feedback_training.py --data path/to.csv --target col --query "..."

At the model_selection / training checkpoint prompts, either:
    - press Enter (or type "accept")  -> approves and continues
    - type any other text             -> sent as feedback_text:
        - at model_selection: model_selection_agent re-plans directly
        - at training:        model_selection_agent re-plans, then
                               training_agent retrains with the new plan
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

INTERACTIVE_AGENTS = {"model_selection", "training"}
DEFAULT_DATA   = str(PROJECT_ROOT / "assets/data/Classification Datasets/Titanic-Dataset.csv")
DEFAULT_TARGET = "Survived"
DEFAULT_QUERY  = (
    "Run EDA, preprocessing, model selection, and training on this dataset. "
    "Use the simple training approach with RandomForest by default."
)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive feedback test for the Model Selection <-> Training loop"
    )
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
    run_id = f"test-feedback-training-{os.urandom(3).hex()}"

    print("=" * 70)
    print("FEEDBACK TEST — MODEL SELECTION <-> TRAINING LOOP")
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

    plan_runs = []      # snapshots of agent_outputs["model_selection"] over time
    training_runs = []  # snapshots of agent_outputs["training"] over time
    iteration, max_steps = 0, 30

    while state.get("__interrupted__") and iteration < max_steps:
        paused_at = state["__paused_at__"]
        output = state.get("agent_outputs", {}).get(paused_at, {})

        print("\n" + "-" * 70)
        print(f"[PAUSED] checkpoint: '{paused_at}'")
        print("-" * 70)

        if paused_at == "model_selection":
            plan_runs.append(output)
            print(f"train_tool   : {output.get('train_tool')}")
            print(f"plan_preview : {json.dumps(output.get('plan_preview'), indent=2, default=str)[:1500]}")
        elif paused_at == "training":
            training_runs.append(output)
            print(json.dumps(output, indent=2, default=str)[:1500])

        if paused_at in INTERACTIVE_AGENTS:
            print("-" * 70)
            choice = input(
                f"Enter feedback for '{paused_at}' (Enter / 'accept' to approve): "
            ).strip()
            if choice and choice.lower() != "accept":
                decision, feedback_text = "feedback", choice
                if paused_at == "training":
                    print(f"[SENDING FEEDBACK] '{feedback_text}' -> rerouting to model_selection_agent to re-plan")
                else:
                    print(f"[SENDING FEEDBACK] '{feedback_text}' -> model_selection_agent re-plans")
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

        print(f"\n[PLAN HISTORY] {len(plan_runs)} model_selection run(s):")
        for i, p in enumerate(plan_runs, 1):
            print(f"   {i}. train_tool={p.get('train_tool')}")

        print(f"\n[TRAINING HISTORY] {len(training_runs)} training run(s):")
        for i, t in enumerate(training_runs, 1):
            print(f"   {i}. best_model={t.get('best_model')}  best_score={t.get('best_score')}  status={t.get('status')}")

        history = [h for h in state.get("feedback_history", []) if h.get("agent") in INTERACTIVE_AGENTS]
        if history:
            print(f"\n[OK] feedback_history has {len(history)} entr(y/ies):")
            for h in history:
                print(f"   - agent={h.get('agent')} iteration={h.get('iteration')}: {h.get('feedback_text')}")
        else:
            print("\n[INFO] No feedback was given this run (you accepted immediately each time).")

        if len(training_runs) > 1:
            changed = training_runs[0].get("best_model") != training_runs[-1].get("best_model")
            print(f"\n[{'OK' if changed else 'INFO'}] best_model {'changed' if changed else 'did NOT change'} "
                  f"between first ({training_runs[0].get('best_model')}) and last ({training_runs[-1].get('best_model')}) training run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
