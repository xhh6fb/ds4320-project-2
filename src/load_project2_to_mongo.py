from pymongo import MongoClient
import json
from utils_project2 import setup_logger

# -----------------------------------------
# SETUP LOGGER
# -----------------------------------------
logger = setup_logger("mongo_load.log")

try:
    logger.info("Connecting to MongoDB")

    client = MongoClient("mongodb+srv://<username>:<password>@cluster0.6mite1b.mongodb.net/?appName=Cluster0")

    db = client["project2_db"]
    collection = db["qb_games"]

    # -----------------------------------------
    # LOAD JSON FILE
    # -----------------------------------------
    with open("data/qb_documents.json") as f:
        docs = json.load(f)

    logger.info(f"Loaded {len(docs)} documents")

    # -----------------------------------------
    # INSERT INTO DATABASE
    # -----------------------------------------
    collection.delete_many({})
    collection.insert_many(docs)

    logger.info("Data successfully inserted into MongoDB")

except Exception as e:
    logger.error("MongoDB ERROR")
    logger.error(str(e))
    raise
