from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("MONGO_DB")]
reports_collection = db["reports"]

def store_initial_knowledge_graph( state: dict) -> list:
    """
    Called after IntentDetectorAgent.
    Saves the selected workflow stages to MongoDB.
    """

    knowledge_graph = [
        flag
        for flag, value in state.get("intent_flags", {}).items()
        if value
    ]

    # drop only the last selected flag and keep the others
    knowledge_graph = knowledge_graph[:-1]

    run_id = "6a395d519b0c281d5031ad4f"  # Replace with actual run_id if available

    if run_id:
        try:
            query = (
                {"_id": ObjectId(run_id)}
                if ObjectId.is_valid(run_id)
                else {"run_id": run_id}
            )

            reports_collection.update_one(
                query,
                {
                    "$set": {
                        "knowledge_graph": knowledge_graph,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
                upsert=False,
            )

            print(f"[KnowledgeGraph] Saved: {knowledge_graph}")

        except Exception as e:
            print(f"[KnowledgeGraph] MongoDB error: {e}")

    return knowledge_graph