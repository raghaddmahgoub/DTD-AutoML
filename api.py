import json
import asyncio
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from orchestrator import DTDPipeline
app = FastAPI()
pipeline_instance = DTDPipeline()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# @app.post("/run-pipeline")
# async def run_pipeline(request: Request):
#     body = await request.json()
    
#     # Initial state inputs from the user
#     inputs = {
#         "data_path": body.get("data_path"),
#         "target_column": body.get("target_column"),
#         "task_type": body.get("task_type", "classification")
#     }

#     async def event_generator():
#         # .stream() yields updates after every node execution
#         for output in pipeline_instance.workflow.stream(inputs):
#             # output is a dict like: {'node_name': {updated_state_keys}}
#             for node_name, state_update in output.items():
#                 payload = {
#                     "agent": node_name,
#                     "output": state_update.get("agent_output"),
#                     "error": state_update.get("error")
#                 }

#                 yield f"data: {json.dumps(payload)}\n\n"
#             # Small sleep to ensure the event loop yields to the network
#             await asyncio.sleep(0.1)
        
#         yield "data: {\"status\": \"completed\"}\n\n"

#     return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/run-pipeline")
async def run_pipeline(
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
        "task_type": task_type
    }

    async def event_generator():
        for output in pipeline_instance.workflow.stream(inputs):
            for node_name, state_update in output.items():
                payload = {
                    "agent": node_name,
                    "output": state_update.get("agent_output"),
                    "error": state_update.get("error")
                }

                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0.05)

        yield "data: {\"status\": \"completed\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)