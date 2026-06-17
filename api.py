import os
import json
import asyncio
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from bson import ObjectId
from pymongo import MongoClient

from orchestrator import DTDPipeline
from dotenv import load_dotenv
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
# Load environment variables
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
from fastapi.responses import FileResponse
# Create a global executor for heavy lifting
executor = ThreadPoolExecutor(max_workers=5)

import pandas as pd
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from agents.static.eda_agent.eda_agent import TargetSuggestionAgent

@app.post("/suggest-target")
async def suggest_target(file: UploadFile = File(...)):
    print("Received file:", file.filename if file else "NO FILE")
    try:
        # Save temporarily (optional)
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load dataset
        if file.filename.endswith(".csv"):
            df = pd.read_csv(temp_path)
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(temp_path)
        elif file.filename.endswith(".json"):
            df = pd.read_json(temp_path)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": "Unsupported file format"}
            )

        agent = TargetSuggestionAgent(df)
        result = agent.run()

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/run-pipeline/{dataset_id}/{report_id}")
async def run_pipeline(
    dataset_id: str,
    report_id: str,
    file: UploadFile = File(...),
    target_column: str = Form(...),
):
    # Save uploaded file
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
        start_recorded = False  # flag to record start_time only once
        def get_stream():
            return pipeline_instance.workflow.stream(inputs)

        def safe_next(gen):
            try:
                return next(gen)
            except StopIteration:
                return None
        # Offload the blocking generator to the executor
        # We wrap it to ensure each 'yield' from LangGraph can be awaited by FastAPI
        gen = await loop.run_in_executor(executor, get_stream)
        while True:
            try:
                # Get next item WITHOUT blocking event loop
                output = await loop.run_in_executor(None, safe_next, gen)

            except StopIteration:
                break
        # for output in pipeline_instance.workflow.stream(inputs):
            if output is None:
                break
            for node_name, state_update in output.items():
                # Record start_time only once at first node
                print("Streaming:", node_name)
                if not start_recorded:
                    start_time = datetime.utcnow()
                    reports_collection.update_one(
                        {"_id": ObjectId(report_id)},
                        {"$set": {"start_time": start_time}}
                    )
                    start_recorded = True

                payload = {
                    "agent": node_name,
                    "output": state_update.get("agent_output"),
                    "error": state_update.get("error"),
                    "datasetId": dataset_id,
                    "reportId": report_id,
                }

                # Incremental Mongo update
                if state_update.get("agent_output") is not None:
                    reports_collection.update_one(
                        {"_id": ObjectId(report_id)},
                        {"$set": {f"report.{node_name}": state_update["agent_output"]}}
                    )

                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.05)

        # Pipeline finished, compute runtime and store end_time
        end_time = datetime.utcnow()
        report = reports_collection.find_one({"_id": ObjectId(report_id)})
        start_time = report.get("start_time", end_time)

        if isinstance(start_time, int) or isinstance(start_time, float):
            start_time = datetime.utcfromtimestamp(start_time) # fallback if missing
        runtime = (end_time - start_time).total_seconds()

        reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": {"runtime_seconds": runtime, "end_time": end_time}}
        )

        yield f"data: {json.dumps({'status': 'completed', 'reportId': report_id, 'datasetId': dataset_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/run-custom-pipeline/{dataset_id}/{report_id}")
async def run_custom_pipeline(
    dataset_id: str,
    report_id: str,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    prompt: str = Form(...),
):
    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    inputs = {
        "data_path": str(file_path),
        "target_column": target_column,
        "dataset_id": dataset_id,
        "report_id": report_id,
        "prompt": prompt,
    }
    print("Received inputs for custom pipeline:", inputs)
    
if __name__ == "__main__":
    import uvicorn
    print("Starting API server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
