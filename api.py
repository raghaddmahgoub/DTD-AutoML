# import os
# import uuid
# import shutil
# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse

# from orchestrator import DTDPipeline  # wherever your DTDPipeline lives

# UPLOAD_DIR = "uploaded_data"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# app = FastAPI(
#     title="Interactive EDA + AutoML Pipeline",
#     description="Runs EDA and AutoML using LangGraph",
#     version="1.0.0"
# )

# @app.post("/run-pipeline")
# def run_pipeline(
#     dataset: UploadFile = File(...),
#     target_column: str = Form(...),
#     task_type: str = Form(...)
# ):
#     """
#     Run the full LangGraph pipeline on a user-provided dataset.
#     """

#     if not dataset.filename.endswith(".csv"):
#         raise HTTPException(status_code=400, detail="Only CSV files are supported.")

#     # ── Save file safely ─────────────────────────────
#     file_id = str(uuid.uuid4())
#     file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{dataset.filename}")

#     with open(file_path, "wb") as f:
#         shutil.copyfileobj(dataset.file, f)

#     # ── Build initial LangGraph state ─────────────────
#     initial_state = {
#         "data_path": file_path,
#         "target_column": target_column,
#         "task_type": task_type,
#         "analysis_report_path": None,
#         "automl_directives": None,
#         "final_metrics": None
#     }

#     # ── Run pipeline ─────────────────────────────────
#     try:
#         pipeline = DTDPipeline()
#         final_state = pipeline.workflow.invoke(initial_state)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     # ── Response ─────────────────────────────────────
#     return JSONResponse(
#         content={
#             "status": "completed",
#             "target": target_column,
#             "task_type": task_type,
#             "final_metrics": final_state.get("final_metrics"),
#             "errors": final_state.get("error"),
#         }
#     )
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from orchestrator import DTDPipeline

app = FastAPI()
pipeline_instance = DTDPipeline()

@app.post("/run-pipeline")
async def run_pipeline(request: Request):
    body = await request.json()
    
    # Initial state inputs from the user
    inputs = {
        "data_path": body.get("data_path"),
        "target_column": body.get("target_column"),
        "task_type": body.get("task_type", "classification")
    }

    async def event_generator():
        # .stream() yields updates after every node execution
        for output in pipeline_instance.workflow.stream(inputs):
            # output is a dict like: {'node_name': {updated_state_keys}}
            for node_name, state_update in output.items():
                yield f"data: {json.dumps({'agent': node_name, 'update': state_update})}\n\n"
            
            # Small sleep to ensure the event loop yields to the network
            await asyncio.sleep(0.1)
        
        yield "data: {\"status\": \"completed\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)