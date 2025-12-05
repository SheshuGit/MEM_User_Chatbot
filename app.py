"""
Updated app.py optimized for Render deployments.

Key changes:
- Starts Flask immediately on the port provided by Render ($PORT) so Render port-scan succeeds.
- Initializes the heavy RAG pipeline in a background daemon thread to avoid blocking the HTTP port detection.
- Handles missing/old huggingface/sentence-transformers imports gracefully and logs clear actionable errors.
- Keeps your existing endpoints and memory, but ensures the /api/health endpoint reports rag initialization status.
- Uses environment variable PORT (with fallback to 5000) and binds to 0.0.0.0.

Run with a production WSGI server (recommended):
    gunicorn -w 4 -b 0.0.0.0:$PORT "app:app"

Requirements suggestions are provided outside this file (see deployment notes below).
"""

import os
import threading
import traceback
import json
import logging
from typing import List

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient

# langchain imports (keep as-is from your original code)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# dotenv (optional in Render; env vars are preferred)
from dotenv import load_dotenv

# Load environment variables from .env when present (local dev)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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

# Optional local fallback JSON (useful for local dev)
LOCAL_FALLBACK_JSON = os.path.join(os.path.dirname(__file__), "..", "wedding_api.json")

# Flag to indicate RAG initialization state
rag_init_lock = threading.Lock()
rag_initialized = False
rag_init_error = None


def extract_data_from_mongodb() -> List[dict]:
    """
    Extract vendor documents from MongoDB with a short timeout and optional local fallback.
    """
    try:
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
            logger.info("MongoDB returned 0 documents.")
            return []

        logger.info(f"[Mongo] Successfully loaded {len(documents)} documents from {DB_NAME}.{COLLECTION_NAME}")
        return documents

    except Exception as e:
        logger.warning(f"[Mongo] Error extracting data from MongoDB: {e}")
        traceback.print_exc()
        # Attempt fallback to local JSON (if provided)
        try:
            if os.path.exists(LOCAL_FALLBACK_JSON):
                with open(LOCAL_FALLBACK_JSON, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"[Fallback] Loaded {len(data)} documents from local JSON: {LOCAL_FALLBACK_JSON}")
                return data
            else:
                logger.info(f"[Fallback] No local fallback file found at: {LOCAL_FALLBACK_JSON}")
        except Exception as jf:
            logger.warning(f"[Fallback] Error loading local JSON fallback: {jf}")
            traceback.print_exc()

        return []


def prepare_documents_for_rag(vendor_docs: List[dict]) -> List[str]:
    """
    Convert vendor documents to textual documents suitable for embedding and RAG retrieval.
    """
    documents = []

    for vendor in vendor_docs:
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


def initialize_rag_pipeline() -> bool:
    """
    Initialize the RAG pipeline in a background thread. Returns True on success.

    This function will set the global variables vectorstore and qa_chain on success.
    On failure, rag_init_error will be set with the exception text.
    """
    global vectorstore, qa_chain, rag_initialized, rag_init_error

    with rag_init_lock:
        if rag_initialized:
            logger.info("RAG already initialized; skipping.")
            return True

        logger.info("Initializing RAG Pipeline (background)...")

        try:
            vendor_data = extract_data_from_mongodb()

            if not vendor_data:
                raise RuntimeError("No vendor data available for RAG initialization.")

            documents = prepare_documents_for_rag(vendor_data)
            if not documents:
                raise RuntimeError("No textual documents prepared for RAG initialization.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            texts = text_splitter.create_documents(documents)
            if not texts:
                raise RuntimeError("No text chunks created for RAG initialization.")

            # Initialize embeddings
            logger.info("Initializing HuggingFace embeddings (this may download model weights)...")

            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"}
                )
            except Exception as emb_exc:
                # Provide a clear, actionable error to logs so deployer can pin versions
                logger.error("Failed to initialize HuggingFaceEmbeddings: %s", emb_exc)
                raise

            logger.info("Creating FAISS vector store...")
            vectorstore = FAISS.from_documents(texts, embeddings)
            logger.info("Vector store created")

            if not GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not set. Skipping LLM initialization. RAG will be incomplete.")
                raise RuntimeError("GROQ_API_KEY not set")

            logger.info("Initializing Groq LLM...")
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.3-70b-versatile",
                temperature=0.3
            )

            prompt_template = """You are a helpful wedding planning assistant. Use the following context to answer questions about wedding vendors, services, pricing and availability.

Context: {context}

Question: {question}

Provide a helpful, friendly, and detailed answer. If you're recommending vendors, include their contact information if available, services, price/price range, and availability. If you don't know the answer, say so politely.
"""
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": PROMPT}
            )

            rag_initialized = True
            rag_init_error = None
            logger.info("RAG PIPELINE INITIALIZED SUCCESSFULLY")
            return True

        except Exception as e:
            rag_initialized = False
            rag_init_error = str(e)
            logger.error("RAG initialization failed: %s", e)
            traceback.print_exc()
            return False


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check for service. Indicates whether RAG init succeeded."""
    return jsonify({
        "status": "healthy",
        "rag_initialized": rag_initialized,
        "rag_init_error": rag_init_error
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

        if not rag_initialized or qa_chain is None:
            return jsonify({"error": "RAG pipeline not initialized yet"}), 503

        # Ensure memory exists for session
        if session_id not in conversation_memories:
            conversation_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

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
        logger.error("Error in /api/chat: %s", e)
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
        logger.error("Error in /api/reset: %s", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """Return distinct vendor types (previously 'category')"""
    try:
        vendor_docs = extract_data_from_mongodb()
        types = list({(doc.get("type") or "Unknown") for doc in vendor_docs})
        return jsonify({"categories": types})
    except Exception as e:
        logger.error("/api/categories: %s", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def start_background_rag():
    """Helper to start RAG initialization in a daemon thread."""
    thread = threading.Thread(target=initialize_rag_pipeline, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    logger.info("Starting Wedding Vendor Chatbot API (main)...")

    # Start RAG initialization in background so Flask can bind quickly and Render detects the port
    start_background_rag()

    # Determine port (Render exposes $PORT). Fallback to 5000 for local dev
    port = int(os.getenv("PORT", 5000))

    # In production you should use a WSGI server such as gunicorn. This is the dev server.
    app.run(host="0.0.0.0", port=port)
