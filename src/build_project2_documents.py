import pandas as pd
import nflreadpy as nfl
import json

from utils_project2 import *

# -----------------------------------------
# SETUP LOGGER
# -----------------------------------------
logger = setup_logger("build_data.log")

logger.info("STARTING DATA BUILD PROCESS")

try:
    # -----------------------------------------
    # LOAD RAW DATA
    # -----------------------------------------
    logger.info("Loading nflverse play-by-play data")
    pbp = nfl.load_pbp_data()

    # -----------------------------------------
    # FILTER TO PASSING PLAYS ONLY
    # -----------------------------------------
    pbp = pbp[pbp["pass_attempt"] == 1]

    logger.info(f"Remaining rows after filter: {len(pbp)}")

    # -----------------------------------------
    # AGGREGATE TO QB-GAME LEVEL
    # -----------------------------------------
    qb = pbp.groupby(
        ["season", "week", "passer_id", "passer_player_name", "posteam", "defteam"]
    ).agg({
        "passing_yards": "sum",
        "pass_touchdown": "sum",
        "pass_attempt": "sum"
    }).reset_index()

    qb.columns = [
        "season", "week", "player_id", "player_name",
        "team", "opponent",
        "passing_yards", "passing_tds", "attempts"
    ]

    logger.info(f"QB-game rows: {len(qb)}")

    # -----------------------------------------
    # CLEAN DATA
    # remove low-attempt noise
    # -----------------------------------------
    qb = qb[qb["attempts"] >= 5]

    logger.info("Filtered to meaningful QB performances")

    # -----------------------------------------
    # ADD BASIC CONTEXT (simplified)
    # -----------------------------------------
    qb["team"] = "UNK"
    qb["opponent"] = "UNK"
    qb["is_home"] = True
    qb["game_date"] = pd.to_datetime("2024-01-01")

    # -----------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------
    qb = add_rest_days(qb, logger)
    qb = add_rolling_features(qb, logger)

    # -----------------------------------------
    # REMOVE ROWS WITHOUT HISTORY
    # -----------------------------------------
    qb = qb.dropna()

    logger.info(f"Final dataset size: {len(qb)}")

    # -----------------------------------------
    # CREATE DOCUMENTS
    # -----------------------------------------
    documents = [row_to_document(row) for _, row in qb.iterrows()]

    # -----------------------------------------
    # SAVE JSON
    # -----------------------------------------
    with open("data/qb_documents.json", "w") as f:
        json.dump(documents, f)

    logger.info("Documents successfully saved")

except Exception as e:
    logger.error("ERROR OCCURRED")
    logger.error(str(e))
    raise
