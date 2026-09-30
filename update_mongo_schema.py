# update_mongodb_schema.py
from pymongo import MongoClient
from datetime import datetime
import os

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["outbound_calls"]
collection = db["call_queue"]

# Add new fields to existing documents
collection.update_many(
    {},
    {
        "$set": {
            "retry_after": None,
            "callback_time": None,
            "callback_requested_by": None,
            "best_time_to_call": None,
            "call_duration": 0,
            "outcome": None,
            "last_failure_reason": None
        },
        "$setOnInsert": {
            "call_attempts": 0,
            "created_at": datetime.utcnow()
        }
    },
    upsert=False
)

print("✅ MongoDB schema updated!")
