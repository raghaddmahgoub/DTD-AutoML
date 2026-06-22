import os
import json
import asyncio
import shutil
from agents.dynamic.controller_agent.controller_agent import ControllerAgent
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from bson import ObjectId
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
from src.utils.logger import Logger
from agents.static.orchestrator import DTDPipeline
from dotenv import load_dotenv
from agents.static.eda_agent.eda_agent import TargetSuggestionAgent

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

app = FastAPI()
pipeline_instance = DTDPipeline()

# --- Upload directory ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Mongo connection ---
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
reports_collection = db["reports"]
print("Connected to MongoDB database:", MONGO_DB)

executor = ThreadPoolExecutor(max_workers=5)

# ── Dynamic pipeline (LangGraph + HITL) ──────────────────────────────────────
# Single shared instance so MemorySaver persists state across HTTP requests.
# run_id (== report_id) is the thread_id key inside MemorySaver.
_dynamic_controller = ControllerAgent()


def _sanitize_json_values(val):
    """Recursively replace non-JSON-compliant floats (NaN, Inf) with None."""
    import math
    if isinstance(val, dict):
        return {k: _sanitize_json_values(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_sanitize_json_values(v) for v in val]
    elif isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val

def _dynamic_state_to_response(state: dict, run_id: str) -> dict:
    """
    Normalise the raw PipelineState / ControllerAgent return value into a
    clean JSON response the Node.js backend can parse.

    Possible shapes:
        status="paused"    — HITL checkpoint fired; frontend should show agent
                             output and call /dynamic/resume/{run_id}
        status="completed" — pipeline finished successfully
        status="error"     — unhandled exception inside the graph
    """
    if state.get("__error__"):
        res = {
            "run_id": run_id,
            "status": "error",
            "error":  state["__error__"],
        }
    elif state.get("__interrupted__"):
        paused_at   = state.get("__paused_at__", "unknown")
        agent_out   = state.get("agent_outputs", {}).get(paused_at, {})
        res = {
            "run_id":       run_id,
            "status":       "paused",
            "paused_at":    paused_at,
            # The full output of the just-completed agent, ready for rendering
            "agent_output": agent_out,
        }
    else:
        res = {
            "run_id": run_id,
            "status": "completed",
            "result": {
                "target_column":      state.get("target_column"),
                "task_type":          state.get("task_type"),
                "trained_model_path": state.get("trained_model_path"),
                "model_metrics":      state.get("model_metrics"),
                "endpoint_url":       state.get("endpoint_url"),
                # All per-agent outputs for the frontend panels
                "agent_outputs":      state.get("agent_outputs", {}),
            },
        }
    return _sanitize_json_values(res)

def _dynamic_persist_to_mongo(report_id: str, state: dict) -> None:
    """Mirror dynamic pipeline state fields into the existing MongoDB document."""
    try:
        agent_outputs = state.get("agent_outputs", {})
        status = (
            "paused"    if state.get("__interrupted__") else
            "error"     if state.get("__error__")       else
            "completed"
        )

        # Sanitize full pipeline state for MongoDB persistence to avoid non-BSON errors
        try:
            serialized_state = json.loads(json.dumps(state, default=str))
        except Exception as e:
            print(f"[API][dynamic] State serialization error: {e}")
            serialized_state = state

        update = {
            "updated_at":    datetime.utcnow(),
            "target_column": state.get("target_column"),
            "task_type":     state.get("task_type"),
            "dynamic_status": status,
            "pipeline_state": serialized_state,
        }
        # Store each agent's output under report.<agent_name>
        for agent_name, output in agent_outputs.items():
            update[f"report.{agent_name}"] = output

        reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": update},
            upsert=True,
        )
    except Exception as exc:
        print(f"[API][dynamic] MongoDB persist error: {exc}")

@app.post("/suggest-target")
async def suggest_target(file: UploadFile = File(...)):
    print("Received file:", file.filename if file else "NO FILE")
    try:
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load dataset
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            df = pd.read_csv(temp_path)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(temp_path)
        elif ext == "json":
            df = pd.read_json(temp_path)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})

        result = TargetSuggestionAgent(df).run()
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/run-pipeline/{dataset_id}/{report_id}")
async def run_pipeline(
    dataset_id: str,
    report_id: str,
    file: UploadFile = File(...),
    target_column: str = Form(...),
):
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    inputs = {
        "data_path": str(file_path),
        "target_column": target_column,
        "dataset_id": dataset_id,
        "report_id": report_id,
    }

    async def event_generator():
        loop = asyncio.get_event_loop()
        start_recorded = False

        # A queue bridges the blocking pipeline thread and the async SSE loop.
        # The pipeline thread puts one event dict per stage; None signals done.
        queue: asyncio.Queue = asyncio.Queue()

        def run_pipeline_in_thread():
            """Runs entirely in a ThreadPoolExecutor worker."""
            try:
                for event in pipeline_instance.stream_stages(inputs):
                    # Put the event onto the async queue from the sync thread
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"node_name": "__error__", "agent_output": None,
                     "error": str(exc), "cache_hit": False, "_state": {}},
                )
            finally:
                # Sentinel: signals the async consumer that the pipeline is done
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Fire the pipeline in the background — don't await it here
        executor.submit(run_pipeline_in_thread)

        # Consume events as they arrive; each await unblocks the SSE flush
        while True:
            event = await queue.get()   # blocks only until the next stage finishes
            if event is None:
                break                   # sentinel → pipeline done

            node_name = event["node_name"]
            agent_output = event["agent_output"]
            error = event["error"]
            cache_hit = event["cache_hit"]

            print(f"Streaming: {node_name}  (cache_hit={cache_hit})")

            # Record start_time once
            if not start_recorded:
                reports_collection.update_one(
                    {"_id": ObjectId(report_id)},
                    {"$set": {"start_time": datetime.utcnow()}},
                )
                start_recorded = True

            # Mirror to Mongo (skip the cache_check banner — it has no output)
            if agent_output is not None:
                reports_collection.update_one(
                    {"_id": ObjectId(report_id)},
                    {"$set": {f"report.{node_name}": agent_output}},
                )

            payload = {
                "agent": node_name,
                "output": agent_output,
                "error": error,
                "cache_hit": cache_hit,
                "datasetId": dataset_id,
                "reportId":  report_id,
            }
            yield f"data: {json.dumps(payload, default=str)}\n\n"
            # No sleep needed — queue.get() already yields control each iteration

        end_time = datetime.utcnow()
        report   = reports_collection.find_one({"_id": ObjectId(report_id)})
        s_time   = report.get("start_time", end_time)
        if isinstance(s_time, (int, float)):
            s_time = datetime.utcfromtimestamp(s_time)

        runtime = (end_time - s_time).total_seconds()
        reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": {"runtime_seconds": runtime, "end_time": end_time}},
        )
        yield f"data: {json.dumps({'status': 'completed', 'reportId': report_id, 'datasetId': dataset_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/run-custom-pipeline/{dataset_id}/{report_id}")
async def run_custom_pipeline(
    dataset_id:    str,
    report_id:     str,
    file:          UploadFile = File(...),
    target_column: str        = Form(...),
    prompt:        str        = Form(...),
):
    file_path = UPLOAD_DIR / file.filename
    print("Inputs received api.py:", dataset_id, report_id, file.filename, target_column, prompt)
    print("Target column:", target_column)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    inputs = {
        "data_path":     str(file_path),
        "target_column": target_column,
        "dataset_id":    dataset_id,
        "report_id":     report_id,
        "prompt":        prompt,
    }
    loop = asyncio.get_event_loop()

    def _run():
        return _dynamic_controller.run(inputs)

    print("Received inputs for custom pipeline:", inputs)
    result = await loop.run_in_executor(executor, _run)

    _dynamic_persist_to_mongo(report_id, result)

    return {
        "status": "success",
        "result": result,
    }

@app.post("/dynamic/run/{report_id}")
async def dynamic_run_pipeline(
    report_id:     str,
    file:          UploadFile = File(...),
    prompt:        str        = Form(...),
    target_column: str        = Form(None),  # optional — Intent Detector infers if absent
):
    """
    Start a new dynamic LangGraph pipeline run.

    - report_id is used as the LangGraph thread_id (MemorySaver key).
    - target_column is optional; the Intent Detector will suggest one if omitted.
    - Returns immediately: status is "paused" (HITL checkpoint) or "completed".

    Frontend flow after receiving status=="paused":
        1. Render agent_output to the user
        2. Collect decision ("accept" / "feedback") + optional feedback_text
        3. POST /dynamic/resume/{run_id}
    """
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Record run start in Mongo
    reports_collection.update_one(
        {"_id": ObjectId(report_id)},
        {"$set": {"start_time": datetime.utcnow(), "dynamic_status": "running"}},
        upsert=True,
    )

    loop = asyncio.get_event_loop()

    def _run():
        return _dynamic_controller.run({
            "data_path":     str(file_path),
            "prompt":        prompt,
            "target_column": target_column,   # None → Intent Detector will infer
            "report_id":     report_id,        # doubles as thread_id in MemorySaver
        })

    state = await loop.run_in_executor(executor, _run)

    _dynamic_persist_to_mongo(report_id, state)
    return JSONResponse(content=_dynamic_state_to_response(state, run_id=report_id))

@app.post("/dynamic/resume/{run_id}")
async def dynamic_resume_pipeline(
    run_id:        str,
    decision:      str = Form(...),   # "accept" | "feedback"
    feedback_text: str = Form(""),    # only used when decision == "feedback"
):
    """
    Resume a paused HITL checkpoint.

    Form fields:
        decision      — "accept" (approve and continue) | "feedback" (re-run agent)
        feedback_text — free-text note for the agent (required when decision=="feedback")

    Returns the same JSON shape as /dynamic/run — may pause again at the next
    checkpoint, or complete if this was the last one.
    """
    loop = asyncio.get_event_loop()

    # Before invoking resume, check if checkpointer has session in memory.
    # If not, retrieve persisted state from MongoDB and re-seed the checkpointer.
    config = {"configurable": {"thread_id": run_id}}
    try:
        snapshot = _dynamic_controller.app.get_state(config)
    except Exception:
        snapshot = None

    if not snapshot or not snapshot.values or "data_path" not in snapshot.values:
        print(f"[API] Session '{run_id}' not found in MemorySaver checkpointer. Restoring from MongoDB...")
        try:
            doc = reports_collection.find_one({"_id": ObjectId(run_id)})
            if doc and "pipeline_state" in doc:
                saved_state = doc["pipeline_state"]
                paused_at = saved_state.get("__paused_at__")
                if paused_at:
                    as_node = f"{paused_at}_agent"
                    # Remove Mongo-specific keys to avoid write issue
                    saved_state.pop("_id", None)
                    _dynamic_controller.app.update_state(config, saved_state, as_node=as_node)
                    print(f"[API] Successfully restored checkpointer state for run_id '{run_id}' as node '{as_node}'")
                else:
                    print(f"[API] Warning: Saved pipeline_state does not contain '__paused_at__'")
            else:
                print(f"[API] Warning: No pipeline state doc found in MongoDB for run_id '{run_id}'")
        except Exception as err:
            print(f"[API] MongoDB state recovery error: {err}")

    def _resume():
        return _dynamic_controller.resume(
            run_id=run_id,
            decision=decision,
            feedback_text=feedback_text,
        )

    state = await loop.run_in_executor(executor, _resume)

    _dynamic_persist_to_mongo(run_id, state)
    return JSONResponse(content=_dynamic_state_to_response(state, run_id=run_id))

@app.get("/dynamic/status/{run_id}")
async def dynamic_pipeline_status(run_id: str):
    """
    Poll the current PipelineState snapshot from MemorySaver.

    Useful if the frontend loses the response from /run or /resume,
    or wants to confirm the latest agent output before rendering.
    """
    try:
        snapshot = _dynamic_controller.app.get_state(
            {"configurable": {"thread_id": run_id}}
        )
        if snapshot is None or not snapshot.values:
            # Fallback to MongoDB if not in memory
            doc = reports_collection.find_one({"_id": ObjectId(run_id)})
            if doc and "pipeline_state" in doc:
                saved_state = doc["pipeline_state"]
                return JSONResponse(content=_dynamic_state_to_response(saved_state, run_id=run_id))
            
            return JSONResponse(
                status_code=404,
                content={"error": f"run_id '{run_id}' not found in memory or database"}
            )
        state = dict(snapshot.values) if snapshot else {}
        return JSONResponse(content=_dynamic_state_to_response(state, run_id=run_id))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == "__main__":
    import uvicorn
    print("Starting API server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)