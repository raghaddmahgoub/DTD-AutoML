import os
import json
import asyncio
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from bson import ObjectId
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

from orchestrator import DTDPipeline
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
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    inputs = {
        "data_path":     str(file_path),
        "target_column": target_column,
        "dataset_id":    dataset_id,
        "report_id":     report_id,
        "prompt":        prompt,
    }
    print("Received inputs for custom pipeline:", inputs)


if __name__ == "__main__":
    import uvicorn
    print("Starting API server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)