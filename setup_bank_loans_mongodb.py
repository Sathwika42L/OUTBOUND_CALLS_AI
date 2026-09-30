"""
Setup MongoDB for Bank Loan Outbound Calls
Creates sample data for loan offers
"""

import os
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = "bank_outbound_calls"
MONGO_COLLECTION = "loan_offers"

def setup_bank_loans_database():
    """Create fresh database with bank loan offers"""
    
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION]
    
    # Drop existing collection to start fresh
    collection.drop()
    print("🗑️  Dropped existing collection")
    
    # Sample loan offers for different customers
    loan_offers = [
        {
            "phone_number": "+27123456789",
            "customer_name": "Rajesh Kumar",
            "loan_type": "Personal Loan",
            "loan_amount": "R 50,000",
            "interest_rate": "10.5% per annum",
            "tenure": "3 years",
            "monthly_emi": "R 1,615",
            "special_offer": "Pre-approved offer with zero processing fees",
            "benefits": "Instant approval, flexible repayment, no collateral needed",
            "urgency": "This special rate is valid only till end of this month",
            "status": "pending",
            "priority": "high",
            "call_attempts": 0,
            "created_at": datetime.utcnow(),
            "target_call_time": datetime.utcnow()
        },
        {
            "phone_number": "+27987654321",
            "customer_name": "Priya Sharma",
            "loan_type": "Home Loan",
            "loan_amount": "R 2,000,000",
            "interest_rate": "8.75% per annum",
            "tenure": "20 years",
            "monthly_emi": "R 17,280",
            "special_offer": "Special discount of 0.5% on interest rate for first 2 years",
            "benefits": "Tax benefits up to R 2 lakhs, quick disbursement, doorstep service",
            "urgency": "Interest rates are expected to go up next quarter",
            "status": "pending",
            "priority": "high",
            "call_attempts": 0,
            "created_at": datetime.utcnow(),
            "target_call_time": datetime.utcnow()
        },
        {
            "phone_number": "+27111222333",
            "customer_name": "Amit Patel",
            "loan_type": "Car Loan",
            "loan_amount": "R 300,000",
            "interest_rate": "9.25% per annum",
            "tenure": "5 years",
            "monthly_emi": "R 6,260",
            "special_offer": "Zero down payment offer for pre-qualified customers",
            "benefits": "Quick approval in 24 hours, covers insurance, on-road registration included",
            "urgency": "Festival season special offer ending soon",
            "status": "pending",
            "priority": "medium",
            "call_attempts": 0,
            "created_at": datetime.utcnow(),
            "target_call_time": datetime.utcnow()
        },
        {
            "phone_number": "+27444555666",
            "customer_name": "Sunita Reddy",
            "loan_type": "Business Loan",
            "loan_amount": "R 500,000",
            "interest_rate": "11% per annum",
            "tenure": "5 years",
            "monthly_emi": "R 10,870",
            "special_offer": "First 6 months interest-free for new businesses",
            "benefits": "No collateral up to R 5 lakhs, business advisory support, quick disbursement",
            "urgency": "Limited slots available for this special scheme",
            "status": "pending",
            "priority": "high",
            "call_attempts": 0,
            "created_at": datetime.utcnow(),
            "target_call_time": datetime.utcnow()
        },
        {
            "phone_number": "+27777888999",
            "customer_name": "Vikram Singh",
            "loan_type": "Education Loan",
            "loan_amount": "R 800,000",
            "interest_rate": "9% per annum",
            "tenure": "10 years",
            "monthly_emi": "R 10,135",
            "special_offer": "Study now, pay later - repayment starts after course completion",
            "benefits": "Covers tuition, living expenses, no collateral needed, tax benefits available",
            "urgency": "Admission season special - apply before deadlines",
            "status": "pending",
            "priority": "medium",
            "call_attempts": 0,
            "created_at": datetime.utcnow(),
            "target_call_time": datetime.utcnow()
        }
    ]
    
    # Insert sample data
    result = collection.insert_many(loan_offers)
    print(f"✅ Inserted {len(result.inserted_ids)} loan offers")
    
    # Create indexes
    collection.create_index("phone_number")
    collection.create_index("status")
    collection.create_index("priority")
    collection.create_index("target_call_time")
    print("✅ Created indexes")
    
    # Show summary
    print("\n📊 Database Summary:")
    print(f"   Database: {MONGO_DB_NAME}")
    print(f"   Collection: {MONGO_COLLECTION}")
    print(f"   Total offers: {collection.count_documents({})}")
    print(f"   Pending: {collection.count_documents({'status': 'pending'})}")
    print(f"   High priority: {collection.count_documents({'priority': 'high'})}")
    
    print("\n📋 Sample Offers:")
    for i, offer in enumerate(collection.find().limit(5), 1):
        print(f"\n{i}. {offer['customer_name']} - {offer['loan_type']}")
        print(f"   Amount: {offer['loan_amount']} @ {offer['interest_rate']}")
        print(f"   EMI: {offer['monthly_emi']} for {offer['tenure']}")
        print(f"   Special: {offer['special_offer']}")
    
    client.close()
    print("\n✅ Database setup complete!")
    print(f"\n🔗 Connection: {MONGO_URI}")
    print(f"📦 Database: {MONGO_DB_NAME}")
    print(f"📁 Collection: {MONGO_COLLECTION}")


def view_offers():
    """View all current loan offers"""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION]
    
    print("\n📞 Current Loan Offers:\n")
    for i, offer in enumerate(collection.find(), 1):
        print(f"{i}. {offer['customer_name']} ({offer['phone_number']})")
        print(f"   Type: {offer['loan_type']} - {offer['loan_amount']}")
        print(f"   Status: {offer['status']} | Priority: {offer['priority']}")
        print(f"   Attempts: {offer['call_attempts']}")
        print()
    
    client.close()


def reset_all_pending():
    """Reset all offers to pending status"""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION]
    
    result = collection.update_many(
        {},
        {
            "$set": {
                "status": "pending",
                "call_attempts": 0
            }
        }
    )
    
    print(f"✅ Reset {result.modified_count} offers to pending")
    client.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "view":
            view_offers()
        elif sys.argv[1] == "reset":
            reset_all_pending()
        else:
            print("Usage:")
            print("  python setup_bank_loans_mongodb.py        # Setup fresh database")
            print("  python setup_bank_loans_mongodb.py view   # View current offers")
            print("  python setup_bank_loans_mongodb.py reset  # Reset all to pending")
    else:
        setup_bank_loans_database()
