from pymongo import MongoClient
import json
from utils_project2 import setup_logger

logger = setup_logger("mongo_load.log")

try:
    logger.info("Connecting to MongoDB")

    client = MongoClient("mongodb+srv://xhh6fb:xv63vmjk@cluster0.6mite1b.mongodb.net/?appName=Cluster0")

    db = client["project2_db"]
    collection = db["qb_games"]

    with open("data/qb_documents.json") as f:
        docs = json.load(f)

    logger.info(f"Loaded {len(docs)} docs")

    collection.delete_many({})
    collection.insert_many(docs)

    logger.info("Upload complete")

except Exception as e:
    logger.error(str(e))
    raise
