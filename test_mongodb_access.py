#!/usr/bin/env python3
"""
Test MongoDB Access - Verify you can read/write to MongoDB
Run this to test database connection and queries
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "outbound_calls"
COLLECTION_NAME = "call_queue"


def test_connection():
    """Test MongoDB connection"""
    print("\n🔌 Testing MongoDB connection...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        print("✅ Connected successfully!")
        return client
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None


def test_get_random_pending_call(collection):
    """Test getting a random pending call (bot's main use case)"""
    print("\n📞 Getting random pending call...")
    
    pipeline = [
        {"$match": {"status": "pending"}},
        {"$sample": {"size": 1}}
    ]
    
    results = list(collection.aggregate(pipeline))
    
    if results:
        call = results[0]
        print("✅ Found pending call:")
        print(f"   Call ID: {call['_id']}")
        print(f"   Phone: {call['phone_number']}")
        print(f"   Name: {call['customer_name']}")
        print(f"   Priority: {call['priority']}")
        print(f"   Message: {call['message'][:80]}...")
        return call
    else:
        print("❌ No pending calls found")
        return None


def test_update_call_status(collection, call_id):
    """Test updating call status (after bot makes call)"""
    print(f"\n📝 Updating call status...")
    
    result = collection.update_one(
        {"_id": call_id},
        {
            "$set": {
                "status": "calling",
                "last_called_at": datetime.utcnow()
            },
            "$inc": {"call_attempts": 1}
        }
    )
    
    if result.modified_count > 0:
        print("✅ Status updated successfully")
        return True
    else:
        print("❌ Update failed")
        return False


def test_mark_completed(collection, call_id):
    """Test marking call as completed"""
    print(f"\n✅ Marking call as completed...")
    
    result = collection.update_one(
        {"_id": call_id},
        {
            "$set": {
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "notes": "Test call completed successfully"
            }
        }
    )
    
    if result.modified_count > 0:
        print("✅ Marked as completed")
        return True
    else:
        print("❌ Failed to mark as completed")
        return False


def test_reset_to_pending(collection, call_id):
    """Reset the test call back to pending status"""
    print(f"\n🔄 Resetting call to pending...")
    
    result = collection.update_one(
        {"_id": call_id},
        {
            "$set": {
                "status": "pending",
                "last_called_at": None,
                "completed_at": None,
                "call_attempts": 0,
                "notes": ""
            }
        }
    )
    
    if result.modified_count > 0:
        print("✅ Reset to pending")
        return True
    else:
        print("❌ Reset failed")
        return False


def show_statistics(collection):
    """Show call queue statistics"""
    print("\n" + "=" * 70)
    print("📊 CALL QUEUE STATISTICS")
    print("=" * 70)
    
    total = collection.count_documents({})
    pending = collection.count_documents({"status": "pending"})
    calling = collection.count_documents({"status": "calling"})
    completed = collection.count_documents({"status": "completed"})
    failed = collection.count_documents({"status": "failed"})
    
    print(f"\nTotal: {total}")
    print(f"Pending: {pending}")
    print(f"Calling: {calling}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    
    if pending > 0:
        print("\n📞 Pending calls by priority:")
        high = collection.count_documents({"status": "pending", "priority": "high"})
        medium = collection.count_documents({"status": "pending", "priority": "medium"})
        low = collection.count_documents({"status": "pending", "priority": "low"})
        print(f"  High: {high}")
        print(f"  Medium: {medium}")
        print(f"  Low: {low}")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("🧪 MONGODB ACCESS TEST")
    print("=" * 70)
    
    # Test 1: Connection
    client = test_connection()
    if not client:
        print("\n❌ Cannot proceed without database connection")
        print("\n💡 Make sure MongoDB is running:")
        print("   Windows: net start MongoDB")
        return
    
    # Get collection
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Test 2: Get random call
    call = test_get_random_pending_call(collection)
    if not call:
        print("\n❌ No pending calls found")
        print("   Run: python setup_sample_mongodb.py")
        return
    
    call_id = call["_id"]
    
    # Test 3: Update status to calling
    test_update_call_status(collection, call_id)
    
    # Test 4: Mark as completed
    test_mark_completed(collection, call_id)
    
    # Test 5: Reset back to pending
    test_reset_to_pending(collection, call_id)
    
    # Show statistics
    show_statistics(collection)
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    
    print("\n🎯 Key operations verified:")
    print("   ✅ Connect to MongoDB")
    print("   ✅ Get random pending call")
    print("   ✅ Update call status")
    print("   ✅ Mark call completed")
    print("   ✅ Query statistics")
    
    print("\n📝 Example code for bot integration:")
    print("""
    # In outbound_bot.py:
    from pymongo import MongoClient
    
    client = MongoClient("mongodb://localhost:27017/")
    collection = client["outbound_calls"]["call_queue"]
    
    # Get random pending call
    call = collection.aggregate([
        {"$match": {"status": "pending"}},
        {"$sample": {"size": 1}}
    ]).next()
    
    # Use call data
    phone = call["phone_number"]
    name = call["customer_name"]
    message = call["message"]
    """)


if __name__ == "__main__":
    run_all_tests()
