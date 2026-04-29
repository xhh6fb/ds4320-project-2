import pandas as pd
import nfl_data_py as nfl
import json
import os

from utils_project2 import *

logger = setup_logger("build_data.log")

logger.info("STARTING DATA BUILD")

try:
    # -----------------------------------------
    # LOAD DATA
    # -----------------------------------------
    pbp = nfl.import_pbp_data([2020, 2021, 2022, 2023, 2024])

    # keep passing plays
    pbp = pbp[pbp["pass_attempt"] == 1]

    # -----------------------------------------
    # QB GAME LEVEL AGGREGATION
    # -----------------------------------------
    qb = pbp.groupby(
        ["season", "week", "passer_id", "passer_player_name", "posteam", "defteam"]
    ).agg(
        passing_yards=("passing_yards", "sum"),
        passing_tds=("pass_touchdown", "sum"),
        attempts=("pass_attempt", "sum")
    ).reset_index()

    qb.rename(columns={
        "passer_id": "player_id",
        "passer_player_name": "player_name",
        "posteam": "team",
        "defteam": "opponent"
    }, inplace=True)

    qb = qb[qb["attempts"] >= 5]

    qb["game_date"] = pd.to_datetime("2024-01-01")  # placeholder (can improve later)

    # -----------------------------------------
    # DEFENSE FEATURES
    # -----------------------------------------
    defense = add_defense_features(pbp, logger)

    qb = qb.merge(
        defense,
        left_on=["season", "week", "opponent"],
        right_on=["season", "week", "team"],
        how="left",
        suffixes=("", "_def")
    )

    # -----------------------------------------
    # ROLLING FEATURES
    # -----------------------------------------
    qb = add_rolling_features(qb, logger)
    qb = add_extended_rolling_features(qb, logger)

    qb = qb.dropna()

    logger.info(f"Final dataset size: {len(qb)}")

    # -----------------------------------------
    # CONVERT TO MONGO DOCS
    # -----------------------------------------
    docs = [row_to_doc(row) for _, row in qb.iterrows()]

    # -----------------------------------------
    # SAVE IN DATA FOLDER
    # -----------------------------------------
    os.makedirs("data", exist_ok=True)
    with open("data/qb_documents.json", "w") as f:
        json.dump(docs, f)

    logger.info("Saved MongoDB documents")

except Exception as e:
    logger.error(str(e))
    raise
