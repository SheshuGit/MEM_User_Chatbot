import os
import json
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME', 'wedding_vendors')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'vendors')

def test_mongodb():
    """Test MongoDB connection"""
    try:
        print("Testing MongoDB connection...")
        client = MongoClient(
            MONGO_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=10000
        )
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        documents = list(collection.find({}))
        print(f"✅ MongoDB connected! Found {len(documents)} documents")
        client.close()
        return documents
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return None

def test_json_fallback():
    """Test JSON file fallback"""
    try:
        print("\nTesting JSON file fallback...")
        json_file = os.path.join(os.path.dirname(__file__), '..', 'wedding_api.js')
        print(f"Looking for file at: {json_file}")
        print(f"File exists: {os.path.exists(json_file)}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ JSON file loaded! Found {len(data)} categories")
        
        # Print first category as sample
        if data:
            print(f"\nSample category: {data[0].get('category', 'Unknown')}")
            print(f"Number of vendors: {len(data[0].get('vendors', []))}")
        
        return data
    except Exception as e:
        print(f"❌ JSON file loading failed: {e}")
        return None

if __name__ == '__main__':
    print("=" * 50)
    print("Testing Data Sources")
    print("=" * 50)
    
    # Test MongoDB
    mongo_data = test_mongodb()
    
    # Test JSON fallback
    json_data = test_json_fallback()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"MongoDB: {'✅ Working' if mongo_data else '❌ Failed'}")
    print(f"JSON Fallback: {'✅ Working' if json_data else '❌ Failed'}")
    print("=" * 50)