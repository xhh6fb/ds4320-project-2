import pandas as pd
import nfl_data_py as nfl
import os

from utils_project2 import *

logger = setup_logger("build.log")

logger.info("STARTING NFL CONTEXT MODEL PIPELINE")

try:
    # ============================================================
    # LOAD DATA
    # ============================================================
    pbp = nfl.import_pbp_data([2020, 2021, 2022, 2023, 2024])

    logger.info(f"Loaded pbp shape: {pbp.shape}")

    # ============================================================
    # FIX DATE TYPE
    # ============================================================

    pbp["game_date"] = pd.to_datetime(pbp["game_date"])

    # ============================================================
    # QB GAME LEVEL DATA
    # ============================================================
    
    qb = pbp.groupby(
        [
            "season",
            "week",
            "game_id",
            "passer_player_id",   # use correct column name
            "passer_player_name",
            "posteam",
            "defteam",
            "home_team",
            "away_team",
            "game_date"
         ]
    ).agg(
        passing_yards=("passing_yards", "sum"),
        pass_attempts=("pass_attempt", "sum"),

        # weather/game context (take first measurement per game)
        temp=("temp", "first"),
        wind=("wind", "first"),
        time_of_day=("time_of_day", "first")
    ).reset_index()

    # ============================================================
    # STANDARDIZE COLUMN NAMES
    # ============================================================
    
    qb.rename(columns={
        "passer_player_id": "player_id",
        "passer_player_name": "player_name",
        "posteam": "team",
        "defteam": "opponent"
    }, inplace=True)

    # ============================================================
    # FEATURE PIPELINE
    # ============================================================
    
    qb = add_qb_form(qb, logger)
    qb = add_rest_days(qb, logger)
    qb = add_weather_features(qb, logger)
    qb = add_home_away(qb, logger)

    # ============================================================
    # DEFENSE FEATURES AND MERGE
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
    logger.info("PIPELINE COMPLETE SUCCESSFULLY")

except Exception as e:
    logger.error("PIPELINE FAILED")
    logger.error(str(e))
    raise
