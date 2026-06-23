from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
reports_collection = db["reports"]

run_id = "6a395d519b0c281d5031ad4f"

def append_agent_output(
    run_id: str,
    agent_name: str,
    output: dict,
) -> None:
    """
    Append a completed agent's output to the agent graph.

    Example stored structure:

    agent_graph = [
        {
            "agent": "eda",
            "output": {...},
            "timestamp": ...
        },
        {
            "agent": "preprocessing",
            "output": {...},
            "timestamp": ...
        }
    ]
    """

    try:
        query = (
            {"_id": ObjectId(run_id)}
            if ObjectId.is_valid(run_id)
            else {"run_id": run_id}
        )

        agent_node = {
            "agent": agent_name,
            "output": output,
            "timestamp": datetime.now(timezone.utc),
        }

        reports_collection.update_one(
            query,
            {
                "$push": {
                    "agent_graph": agent_node
                },
                "$set": {
                    "updated_at": datetime.now(timezone.utc)
                }
            },
            upsert=False,
        )

        print(f"[AgentGraph] Added agent: {agent_name}")

    except Exception as e:
        print(f"[AgentGraph] MongoDB error: {e}")