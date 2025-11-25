"""
Configuration management for the wedding chatbot backend
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    # MongoDB Configuration
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb+srv://Sheshu:Sheshu11@cluster0.ahsalay.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
    DB_NAME = os.getenv('DB_NAME', 'rohith_task')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'wedding_chatbot')
    
    # Groq Configuration
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')
    
    # Embeddings Configuration
    EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDINGS_DEVICE = os.getenv('EMBEDDINGS_DEVICE', 'cpu')
    
    # RAG Configuration
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', '5'))
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.3'))
    
    # Flask Configuration
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is not set")
        
        if not cls.MONGO_URI:
            errors.append("MONGO_URI is not set")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True
    
    @classmethod
    def display(cls):
        """Display current configuration (hiding sensitive data)"""
        print("\n" + "="*50)
        print("CONFIGURATION")
        print("="*50)
        print(f"MongoDB URI: {cls.MONGO_URI[:20]}...")
        print(f"Database: {cls.DB_NAME}")
        print(f"Collection: {cls.COLLECTION_NAME}")
        print(f"Groq Model: {cls.GROQ_MODEL}")
        print(f"Groq API Key: {'*' * 20 if cls.GROQ_API_KEY else 'NOT SET'}")
        print(f"Embeddings Model: {cls.EMBEDDINGS_MODEL}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Retrieval K: {cls.RETRIEVAL_K}")
        print(f"Flask Host: {cls.FLASK_HOST}")
        print(f"Flask Port: {cls.FLASK_PORT}")
        print("="*50 + "\n")

if __name__ == "__main__":
    # Test configuration
    try:
        Config.validate()
        Config.display()
        print("✅ Configuration is valid!")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
