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

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

app = FastAPI()
pipeline_instance = DTDPipeline()

# --- Upload directory ---
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Mongo connection ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]  # now points to your "test" DB in the "AUTH" project

reports_collection = db["reports"]  # collection inside test
print("Connected to MongoDB database:", MONGO_DB)

@app.post("/run-pipeline/{dataset_id}/{report_id}")
async def run_pipeline(
    dataset_id: str,
    report_id: str,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form("classification"),
):
    # Save uploaded file
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    inputs = {
        "data_path": str(file_path),
        "target_column": target_column,
        "task_type": task_type,
        "dataset_id": dataset_id,
        "report_id": report_id,
    }

    async def event_generator():
        for output in pipeline_instance.workflow.stream(inputs):
            for node_name, state_update in output.items():
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

        yield f"data: {json.dumps({'status': 'completed', 'reportId': report_id, 'datasetId': dataset_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)