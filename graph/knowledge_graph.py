from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()

_MONGO_URI = os.getenv("MONGO_URI")
_MONGO_DB  = os.getenv("MONGO_DB")

# MongoDB is optional — when not configured, knowledge-graph storage becomes
# a no-op instead of crashing every module that imports this file
# (intent_detector.py imports it unconditionally).
reports_collection = None
if _MONGO_URI and _MONGO_DB:
    try:
        client = MongoClient(_MONGO_URI)
        db = client[_MONGO_DB]
        reports_collection = db["reports"]
    except Exception as e:
        print(f"[KnowledgeGraph] MongoDB connection failed, storage disabled: {e}")
else:
    print("[KnowledgeGraph] MONGO_URI/MONGO_DB not set — knowledge-graph storage disabled.")


def store_initial_knowledge_graph(state: dict, run_id: str = None) -> list:
    """
    Called after IntentDetectorAgent.
    Saves the selected workflow stages to MongoDB.
    """
    knowledge_graph = [
        flag
        for flag, value in state.get("intent_flags", {}).items()
        if flag.startswith("run_") and value is True
    ]

    if reports_collection is None:
        return knowledge_graph

    if run_id:
        print(f"[KnowledgeGraph] Saving knowledge graph for run_id: {run_id}")
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