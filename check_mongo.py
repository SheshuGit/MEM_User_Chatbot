from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    count = collection.count_documents({})
    print(f"✅ Connected to MongoDB successfully!")
    print(f"Database: {DB_NAME}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents count: {count}")
    
    if count > 0:
        sample = collection.find_one()
        print(f"\nSample document category: {sample.get('type', 'N/A')}")
    else:
        print("\n⚠️ No documents found. You need to import data!")
        
    client.close()
except Exception as e:
    print(f"❌ Error: {e}")