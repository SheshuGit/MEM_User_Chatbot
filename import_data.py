"""
Script to import wedding vendor data from JSON file to MongoDB
"""
import json
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME', 'rohith_task')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'wedding_chatbot')

def import_json_to_mongodb(json_file_path):
    """Import JSON data to MongoDB"""
    try:
        # Read JSON file
        print(f"Reading JSON file: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"Loaded {len(data)} categories from JSON")
        
        # Connect to MongoDB
        print(f"Connecting to MongoDB: {MONGO_URI}")
        client = MongoClient(
            MONGO_URI,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=10000
        )
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Clear existing data (optional - comment out if you want to keep existing data)
        print(f"Clearing existing data in {DB_NAME}.{COLLECTION_NAME}")
        collection.delete_many({})
        
        # Insert data
        print("Inserting data into MongoDB...")
        if isinstance(data, list):
            result = collection.insert_many(data)
            print(f"Successfully inserted {len(result.inserted_ids)} documents")
        else:
            result = collection.insert_one(data)
            print(f"Successfully inserted 1 document")
        
        # Verify insertion
        count = collection.count_documents({})
        print(f"Total documents in collection: {count}")
        
        # Show sample data
        print("\nSample document:")
        sample = collection.find_one()
        if sample:
            print(f"Category: {sample.get('category', 'N/A')}")
            vendors = sample.get('vendors', [])
            print(f"Number of vendors: {len(vendors)}")
            if vendors:
                print(f"First vendor: {vendors[0].get('businessName', 'N/A')}")
        
        client.close()
        print("\n✅ Data import completed successfully!")
        return True
        
    except FileNotFoundError:
        print(f"❌ Error: File not found - {json_file_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    # Default JSON file path (adjust as needed)
    json_file = "../wedding_api.js"
    
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"JSON file not found at: {json_file}")
        json_file = input("Enter the path to your JSON file: ")
    
    import_json_to_mongodb(json_file)
