# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import json
import traceback

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

# Globals
vectorstore = None
qa_chain = None
conversation_memories = {}

# Config from env (fall back to safe defaults)
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "wedding")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vendorservices")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)

# Local fallback file (optional) - adjust path if needed
LOCAL_FALLBACK_JSON = os.path.join(os.path.dirname(__file__), "..", "wedding_api.json")


def extract_data_from_mongodb():
    """
    Extract vendor documents from MongoDB.
    Returns: list of vendor documents (each is a dict).
    """
    try:
        # Use tls=True for Atlas; do NOT disable certificate validation in production.
        client = MongoClient(MONGODB_URI, tls=True, serverSelectionTimeoutMS=5000)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        documents = list(collection.find({}))

        # Convert ObjectId to str for JSON-safety if present
        for d in documents:
            if "_id" in d:
                try:
                    d["_id"] = str(d["_id"])
                except Exception:
                    pass

        client.close()

        if not documents:
            # If the collection is empty, we still return empty list (caller will handle)
            print("MongoDB returned 0 documents.")
            return []

        print(f"[Mongo] Successfully loaded {len(documents)} documents from {DB_NAME}.{COLLECTION_NAME}")
        return documents

    except Exception as e:
        print(f"[Mongo] Error extracting data from MongoDB: {e}")
        traceback.print_exc()
        # Attempt fallback to local JSON (if provided)
        try:
            if os.path.exists(LOCAL_FALLBACK_JSON):
                with open(LOCAL_FALLBACK_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"[Fallback] Loaded {len(data)} documents from local JSON: {LOCAL_FALLBACK_JSON}")
                return data
            else:
                print(f"[Fallback] No local fallback file found at: {LOCAL_FALLBACK_JSON}")
        except Exception as jf:
            print(f"[Fallback] Error loading local JSON fallback: {jf}")
            traceback.print_exc()

        return []


def prepare_documents_for_rag(vendor_docs):
    """
    Convert vendor documents (flat schema per vendor) into textual documents
    suitable for embedding and RAG retrieval.
    Expects vendor_docs to be a list of dicts (each representing one vendor).
    """
    documents = []

    for vendor in vendor_docs:
        # Coerce values to strings and handle arrays elegantly
        name = vendor.get("name") or vendor.get("businessName") or "N/A"
        vtype = vendor.get("type", "N/A")
        subcategory = vendor.get("subcategory", "")
        city = vendor.get("city", vendor.get("cityName", "N/A"))
        location = vendor.get("location", "N/A")
        price = vendor.get("price", "N/A")
        veg_price = vendor.get("vegPrice", "N/A")
        nonveg_price = vendor.get("nonVegPrice", "N/A")
        capacity = vendor.get("capacity", "N/A")
        rooms = vendor.get("rooms", "N/A")
        rental_cost = vendor.get("rentalCost", "N/A")
        services = vendor.get("services", "N/A")
        features = vendor.get("features", []) or []
        features_text = ", ".join(map(str, features)) if features else "N/A"
        image = vendor.get("image", "")
        images = vendor.get("images", []) or []
        images_text = ", ".join(images) if images else (image or "N/A")
        description = vendor.get("description", "")
        availability = vendor.get("availability")
        availability_text = "Available" if availability else "Not Available"

        doc_text = f"""
Vendor Name: {name}
Type: {vtype}
Subcategory: {subcategory}
City: {city}
Location: {location}
Price: {price}
Veg Price: {veg_price}
Non-Veg Price: {nonveg_price}
Capacity: {capacity}
Rooms: {rooms}
Rental Cost: {rental_cost}

Services: {services}
Features: {features_text}

Availability: {availability_text}

Description: {description}
Image/Images: {images_text}
"""
        documents.append(doc_text.strip())

    return documents


def initialize_rag_pipeline():
    """
    Initialize the full RAG pipeline:
      - Load data from MongoDB
      - Prepare text documents
      - Chunk texts
      - Create embeddings (HuggingFace)
      - Build FAISS vectorstore
      - Create ConversationalRetrievalChain around Groq LLM
    Returns True on success, False otherwise.
    """
    global vectorstore, qa_chain

    try:
        print("=" * 60)
        print("Initializing RAG Pipeline...")
        print("=" * 60)
        print("Step 1: Extracting data from MongoDB...")
        vendor_data = extract_data_from_mongodb()

        if not vendor_data:
            print("[ERROR] No vendor data available. Aborting RAG initialization.")
            return False

        print(f"[SUCCESS] Loaded {len(vendor_data)} vendor documents")

        # Prepare documents (text)
        print("Step 2: Preparing documents for RAG...")
        documents = prepare_documents_for_rag(vendor_data)
        if not documents:
            print("[ERROR] No textual documents prepared.")
            return False
        print(f"[SUCCESS] Prepared {len(documents)} textual documents")

        # Text splitting / chunking
        print("Step 3: Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        texts = text_splitter.create_documents(documents)
        if not texts:
            print("[ERROR] No text chunks created.")
            return False
        print(f"[SUCCESS] Created {len(texts)} text chunks")

        # Initialize embeddings
        print("Step 4: Initializing HuggingFace embeddings...")
        print("(This may take 1-2 minutes the first time to download model weights)")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        print("[SUCCESS] Embeddings initialized")

        # Create FAISS vector store
        print("Step 5: Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(texts, embeddings)
        print("[SUCCESS] Vector store created")

        # Initialize Groq LLM (if API key available)
        if not GROQ_API_KEY:
            print("[WARNING] GROQ_API_KEY not set. LLM will not be initialized.")
            return False

        print("Step 6: Initializing Groq LLM...")
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3
        )
        print("[SUCCESS] Groq LLM initialized")

        # Prompt template
        prompt_template = """You are a helpful wedding planning assistant. Use the following context to answer questions about wedding vendors, services, pricing and availability.

Context: {context}

Question: {question}

Provide a helpful, friendly, and detailed answer. If you're recommending vendors, include their contact information if available, services, price/price range, and availability. If you don't know the answer, say so politely.
"""
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create conversational retrieval chain
        print("Step 7: Creating conversational retrieval chain...")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        print("[SUCCESS] Retrieval chain created")
        print("=" * 60)
        print("RAG PIPELINE INITIALIZED SUCCESSFULLY!")
        print("=" * 60)
        return True

    except Exception as e:
        print("[ERROR] ERROR INITIALIZING RAG PIPELINE:", e)
        traceback.print_exc()
        return False


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check for service"""
    return jsonify({
        "status": "healthy",
        "rag_initialized": vectorstore is not None and qa_chain is not None
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """Return conversational response using the RAG pipeline."""
    try:
        payload = request.json or {}
        question = payload.get("question", "").strip()
        session_id = payload.get("session_id", "default")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        if qa_chain is None:
            return jsonify({"error": "RAG pipeline not initialized"}), 500

        # Ensure memory exists for session
        if session_id not in conversation_memories:
            conversation_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

        # Build chain input
        chat_history = conversation_memories[session_id].chat_memory.messages
        result = qa_chain({
            "question": question,
            "chat_history": chat_history
        })

        answer = result.get("answer", "")
        # Save context to memory
        conversation_memories[session_id].save_context(
            {"question": question},
            {"answer": answer}
        )

        return jsonify({
            "answer": answer,
            "session_id": session_id
        })

    except Exception as e:
        print("[ERROR] Error in /api/chat:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset_conversation():
    """Reset conversation memory for a session."""
    try:
        payload = request.json or {}
        session_id = payload.get("session_id", "default")
        if session_id in conversation_memories:
            del conversation_memories[session_id]
        return jsonify({"message": "Conversation reset successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """Return distinct vendor types (previously 'category')"""
    try:
        vendor_docs = extract_data_from_mongodb()
        # Use 'type' field as category/type; fallback to 'Unknown'
        types = list({(doc.get("type") or "Unknown") for doc in vendor_docs})
        return jsonify({"categories": types})
    except Exception as e:
        print("[ERROR] /api/categories:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Wedding Vendor Chatbot API...")
    # Initialize RAG pipeline (will print descriptive errors if it fails)
    ok = initialize_rag_pipeline()
    if not ok:
        print("Warning: RAG pipeline initialization failed. Check your configuration and logs.")
    else:
        print("Server ready to accept requests!")

    # Run Flask dev server (use production WSGI for production)
    app.run(host="0.0.0.0", port=5000, debug=True)
