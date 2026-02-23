import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from orchestrator import DTDPipeline  # wherever your DTDPipeline lives

UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Interactive EDA + AutoML Pipeline",
    description="Runs EDA and AutoML using LangGraph",
    version="1.0.0"
)

@app.post("/run-pipeline")
def run_pipeline(
    dataset: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form(...)
):
    """
    Run the full LangGraph pipeline on a user-provided dataset.
    """

    if not dataset.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    # ── Save file safely ─────────────────────────────
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{dataset.filename}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(dataset.file, f)

    # ── Build initial LangGraph state ─────────────────
    initial_state = {
        "data_path": file_path,
        "target_column": target_column,
        "task_type": task_type,
        "analysis_report_path": None,
        "automl_directives": None,
        "final_metrics": None
    }

    # ── Run pipeline ─────────────────────────────────
    try:
        pipeline = DTDPipeline()
        final_state = pipeline.workflow.invoke(initial_state)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── Response ─────────────────────────────────────
    return JSONResponse(
        content={
            "status": "completed",
            "target": target_column,
            "task_type": task_type,
            "final_metrics": final_state.get("final_metrics"),
            "errors": final_state.get("error"),
        }
    )
