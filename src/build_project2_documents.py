import pandas as pd
import nfl_data_py as nfl
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

    # -----------------------------------------
    # FILTER LOW SAMPLE GAMES
    # -----------------------------------------
    qb = qb[qb["attempts"] >= 5]

    logger.info(f"After aggregation and filtering: {len(qb)} rows")

    # -----------------------------------------
    # APPLY FEATURE ENGINEERING PIPELINE
    # -----------------------------------------
    qb = add_rolling_features(qb, logger)
    qb = add_extended_rolling_features(qb, logger)
    qb = add_defense_features(pbp, qb, logger)

    qb = qb.dropna()

    logger.info(f"Final dataset size: {len(qb)}")

    # -----------------------------------------
    # SAVE FINAL CSV (FOR NOTEBOOK + MONGO + GITHUB)
    # -----------------------------------------
    os.makedirs("data", exist_ok=True)

    output_path = "data/qb_games.csv"
    qb.to_csv(output_path, index=False)

    logger.info(f"Saved dataset to {output_path}")

except Exception as e:
    logger.error("PIPELINE FAILED")
    logger.error(str(e))
    raise
