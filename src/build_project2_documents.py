import pandas as pd
import nfl_data_py as nfl
import os

from utils_project2 import *

logger = setup_logger("build.log")

logger.info("STARTING NFL CONTEXT MODEL PIPELINE")

try:
    # ============================================================
    # LOAD FULL PLAY-BY-PLAY DATA
    # ============================================================
    pbp = nfl.import_pbp_data([2020, 2021, 2022, 2023, 2024])

    # logging
    logger.info(f"Loaded pbp shape: {pbp.shape}")

    # ============================================================
    # QB GAME-LEVEL TARGET DATASET
    # AGGREGATING TARGET VARIABLES
    # ============================================================
    qb = pbp.groupby(
        [
            "season",
            "week",
            "game_id",
            "passer_id",
            "passer_player_name",
            "posteam",
            "defteam",
            "home_team",
            "away_team",
            "game_date",
            "temp",
            "wind",
            "time_of_day"
        ]
    ).agg(
        passing_yards=("passing_yards", "sum"),
        pass_attempts=("pass_attempt", "sum")
    ).reset_index()

    qb.rename(columns={
        "passer_id": "player_id",
        "passer_player_name": "player_name",
        "posteam": "team",
        "defteam": "opponent"
    }, inplace=True)

    # ============================================================
    # FEATURE ENGINEERING (NO QB YARD LEAKAGE)
    # ============================================================

    # QB form (ONLY past info via shift inside function)
    qb = add_qb_form(qb, logger)

    # rest days
    qb = add_rest_days(qb, logger)

    # weather cleanup + binary flag
    qb = add_weather_features(qb, logger)

    # home/away
    qb = add_home_away(qb, logger)

    # ============================================================
    # DEFENSE FEATURES
    # ============================================================
    defense = add_defense_features(pbp, logger)

    qb = qb.merge(
        defense,
        on=["season", "week", "opponent"],
        how="left"
    )

    # ============================================================
    # CLEAN FINAL DATASET
    # ============================================================
    qb = qb.dropna()

    # ============================================================
    # SAVE AS CSV
    # ============================================================
    os.makedirs("data", exist_ok=True)
    qb.to_csv("data/qb_games.csv", index=False)

    logger.info(f"Final dataset shape: {qb.shape}")
    logger.info("PIPELINE COMPLETE")

except Exception as e:
    logger.error("PIPELINE FAILED")
    logger.error(str(e))
    raise
