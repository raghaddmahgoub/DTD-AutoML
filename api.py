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
                payload = {
                    "agent": node_name,
                    "output": state_update.get("agent_output"),
                    "error": state_update.get("error")
                }

                yield f"data: {json.dumps(payload)}\n\n"
            # Small sleep to ensure the event loop yields to the network
            await asyncio.sleep(0.1)
        
        yield "data: {\"status\": \"completed\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)