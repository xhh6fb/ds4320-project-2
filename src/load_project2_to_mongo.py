import json
import logging
from pathlib import Path
import os

from dotenv import load_dotenv
from pymongo import MongoClient

from utils_project2 import setup_logger


def main():
    """
    load the custom qb game documents into mongodb atlas.
    """
    setup_logger("logs/load_project2_to_mongo.log")
    logging.info("starting load_project2_to_mongo.py")

    try:
        load_dotenv()

        mongo_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB", "ds4320_project2")
        collection_name = os.getenv("MONGODB_COLLECTION", "qb_game_docs")

        if not mongo_uri:
            raise ValueError("MONGODB_URI is missing from .env")

        json_path = Path("data/processed/qb_game_documents.json")
        if not json_path.exists():
            raise FileNotFoundError(f"could not find {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            docs = json.load(f)

        logging.info("loaded %s docs from json", len(docs))

        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        # optional: clear existing collection so reruns stay clean
        collection.delete_many({})
        logging.info("cleared existing collection")

        if docs:
            collection.insert_many(docs)
            logging.info("inserted %s documents", len(docs))

        # create a few indexes to make the collection more usable
        collection.create_index("season")
        collection.create_index("week")
        collection.create_index("player_info.player_id")
        collection.create_index("player_info.team")
        collection.create_index("player_info.player_name")

        print(f"done: inserted {len(docs):,} documents into {db_name}.{collection_name}")

    except Exception as e:
        logging.exception("mongo load failed")
        raise e


if __name__ == "__main__":
    main()
