#!/usr/bin/env python3
"""
Simple MongoDB Setup - Create sample outbound call records
Run this once to setup your database with test data
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DATABASE_NAME = "outbound_calls"
COLLECTION_NAME = "call_queue"

# Sample call records
SAMPLE_CALLS = [
    {
        "phone_number": "+27123456789",
        "customer_name": "Rajesh Kumar",
        "message": "your passport application is ready for collection at our Johannesburg office. You can collect it Monday to Friday between nine A M and three P M. Please bring your original ID and the application receipt.",
        "status": "pending",
        "priority": "high",
        "call_attempts": 0,
        "created_at": datetime.utcnow(),
        "last_called_at": None,
        "completed_at": None,
        "notes": ""
    },
    {
        "phone_number": "+27987654321",
        "customer_name": "Priya Sharma",
        "message": "your appointment for Police Clearance Certificate has been scheduled for tomorrow at ten A M. Please bring your passport and proof of address.",
        "status": "pending",
        "priority": "high",
        "call_attempts": 0,
        "created_at": datetime.utcnow(),
        "last_called_at": None,
        "completed_at": None,
        "notes": ""
    },
    {
        "phone_number": "+27111222333",
        "customer_name": "Amit Patel",
        "message": "your visa application has been approved. You can collect your passport with the visa stamp from our office during working hours, Monday to Friday nine A M to three P M.",
        "status": "pending",
        "priority": "medium",
        "call_attempts": 0,
        "created_at": datetime.utcnow(),
        "last_called_at": None,
        "completed_at": None,
        "notes": ""
    },
    {
        "phone_number": "+27444555666",
        "customer_name": "Sunita Reddy",
        "message": "your O C I card application has been processed and is ready for collection. Please bring your old passport and a recent photograph when collecting.",
        "status": "pending",
        "priority": "medium",
        "call_attempts": 0,
        "created_at": datetime.utcnow(),
        "last_called_at": None,
        "completed_at": None,
        "notes": ""
    },
    {
        "phone_number": "+27777888999",
        "customer_name": "Vikram Singh",
        "message": "this is a reminder that your passport renewal documents are incomplete. Please submit the missing birth certificate by end of this week to avoid delays.",
        "status": "pending",
        "priority": "low",
        "call_attempts": 0,
        "created_at": datetime.utcnow(),
        "last_called_at": None,
        "completed_at": None,
        "notes": ""
    }
]


def setup_mongodb():
    """Setup MongoDB with sample outbound call data"""
    
    print("=" * 70)
    print("📞 OUTBOUND CALLS - MongoDB Setup")
    print("=" * 70)
    
    try:
        # Connect to MongoDB
        print(f"\n🔌 Connecting to MongoDB...")
        print(f"   URI: {MONGO_URI}")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        
        # Test connection
        client.server_info()
        print("✅ Connected successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: Cannot connect to MongoDB!")
        print(f"   {str(e)}")
        print("\n💡 Make sure MongoDB is running:")
        print("   Windows: net start MongoDB")
        print("   Or download from: https://www.mongodb.com/try/download/community")
        return
    
    # Get database and collection
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    print(f"\n📦 Database: {DATABASE_NAME}")
    print(f"📋 Collection: {COLLECTION_NAME}")
    
    # Check existing data
    existing_count = collection.count_documents({})
    
    if existing_count > 0:
        print(f"\n⚠️  Collection already has {existing_count} records")
        response = input("   Clear and add fresh sample data? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            collection.delete_many({})
            print("   ✅ Cleared existing records")
        else:
            print("   ℹ️  Keeping existing data")
            print("\n📊 Current records:")
            for i, call in enumerate(collection.find().limit(5), 1):
                print(f"   {i}. {call['customer_name']} - {call['status']} ({call['priority']} priority)")
            return
    
    # Insert sample data
    print(f"\n📝 Inserting {len(SAMPLE_CALLS)} sample records...")
    result = collection.insert_many(SAMPLE_CALLS)
    print(f"✅ Inserted {len(result.inserted_ids)} records")
    
    # Create indexes for performance
    print("\n🔍 Creating indexes...")
    collection.create_index("phone_number")
    collection.create_index("status")
    collection.create_index([("status", 1), ("priority", -1)])
    print("✅ Indexes created")
    
    # Display summary
    print("\n" + "=" * 70)
    print("📊 DATABASE SUMMARY")
    print("=" * 70)
    
    total = collection.count_documents({})
    pending = collection.count_documents({"status": "pending"})
    high = collection.count_documents({"status": "pending", "priority": "high"})
    medium = collection.count_documents({"status": "pending", "priority": "medium"})
    low = collection.count_documents({"status": "pending", "priority": "low"})
    
    print(f"\nTotal records: {total}")
    print(f"Pending calls: {pending}")
    print(f"  - High priority: {high}")
    print(f"  - Medium priority: {medium}")
    print(f"  - Low priority: {low}")
    
    # Show sample records
    print("\n" + "=" * 70)
    print("📞 SAMPLE CALL RECORDS")
    print("=" * 70)
    
    for i, call in enumerate(collection.find({"status": "pending"}), 1):
        print(f"\n{i}. {call['customer_name']} ({call['phone_number']})")
        print(f"   Priority: {call['priority'].upper()}")
        print(f"   Message: {call['message'][:60]}...")
        print(f"   Status: {call['status']}")
    
    print("\n" + "=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    
    print("\n📝 Connection details saved in .env:")
    print(f"   MONGODB_URI={MONGO_URI}")
    print(f"   MONGO_DB_NAME={DATABASE_NAME}")
    print(f"   MONGO_COLLECTION={COLLECTION_NAME}")
    
    print("\n🚀 Next steps:")
    print("   1. Run: python test_mongodb_access.py")
    print("   2. Integrate with outbound_bot.py")
    
    # Update .env if needed
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        if "MONGODB_URI" not in env_content:
            print("\n📝 Adding MongoDB config to .env...")
            with open(env_file, 'a') as f:
                f.write(f"\n# MongoDB Configuration\n")
                f.write(f"MONGODB_URI={MONGO_URI}\n")
                f.write(f"MONGO_DB_NAME={DATABASE_NAME}\n")
                f.write(f"MONGO_COLLECTION={COLLECTION_NAME}\n")
            print("✅ .env updated")


if __name__ == "__main__":
    setup_mongodb()
