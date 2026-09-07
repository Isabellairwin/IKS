import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

_client = None

def get_mongodb_client():
    """
    Establishes and returns a singleton MongoDB client connection.
    Secures the URI by never logging or printing it.
    """
    global _client
    if _client is not None:
        return _client
    
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        # Check if we can find it in parent folder just in case
        raise ValueError("MONGODB_URI environment variable is missing. Please define it in your .env file.")
    
    try:
        # Connect to MongoDB Atlas
        _client = MongoClient(uri)
        # Test connection
        _client.admin.command('ping')
        return _client
    except Exception as e:
        # Clean logging: do not expose uri
        raise RuntimeError(f"Failed to connect to MongoDB. Error details: {str(e)}")

def get_iks_collection():
    """
    Returns the IKS collection handler from the database.
    """
    client = get_mongodb_client()
    db = client["iks_db"]
    collection = db["tamil_to_english_dataset"]
    return collection
