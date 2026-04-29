import pandas as pd
import logging

# -----------------------------------------
# LOGGING SETUP FUNCTION
# This creates a logger that writes to file
# -----------------------------------------
def setup_logger(log_file_name):
    """
    Creates and returns a logger object.
    Logs will be written to the specified file.

    Parameters:
        log_file_name (str): name of the log file

    Returns:
        logger object
    """

    logger = logging.getLogger(log_file_name)

    # prevent duplicate handlers if rerun
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file_name)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


# -----------------------------------------
# SAFE COLUMN ACCESS
# avoids crashing if column missing
# -----------------------------------------
def safe_col(df, col, default=0):
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


# -----------------------------------------
# ADD REST DAYS FEATURE
# -----------------------------------------
def add_rest_days(df, logger):
    logger.info("Adding rest days feature")

    df = df.sort_values(["player_id", "season", "week"])

    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)

    df["days_rest"] = (
        df["game_date"] - df["prev_game_date"]
    ).dt.days

    return df


# -----------------------------------------
# ADD ROLLING FEATURES (NO LEAKAGE)
# -----------------------------------------
def add_rolling_features(df, logger):
    logger.info("Creating rolling features")

    df = df.sort_values(["player_id", "season", "week"])

    # IMPORTANT:
    # shift() ensures we ONLY use past games
    df["avg_pass_yards_last_3"] = (
        df.groupby("player_id")["passing_yards"]
        .transform(lambda x: x.shift().rolling(3).mean())
    )

    df["avg_pass_tds_last_3"] = (
        df.groupby("player_id")["passing_tds"]
        .transform(lambda x: x.shift().rolling(3).mean())
    )

    return df

# -----------------------------------------
# ADD DEFENSE FEATURES
# -----------------------------------------
def add_defense_features(pbp, logger):
    logger.info("Building defensive features")

    defense = pbp.groupby(
        ["season", "week", "defteam"]
    ).agg({
        "passing_yards": "sum",
        "pass_touchdown": "sum",
        "pass_attempt": "sum"
    }).reset_index()

    defense.columns = [
        "season", "week", "team",
        "def_pass_yards",
        "def_pass_tds",
        "def_pass_atts"
    ]

    defense = defense.sort_values(["team", "season", "week"])

    defense["def_yards_pg"] = defense.groupby("team")["def_pass_yards"].transform(lambda x: x.shift().rolling(5).mean())
    defense["def_tds_pg"] = defense.groupby("team")["def_pass_tds"].transform(lambda x: x.shift().rolling(5).mean())

    return defense

# -----------------------------------------
# CONVERT ROW → MONGODB DOCUMENT
# -----------------------------------------
def row_to_doc(row):
    return {
        "_id": f"{row['season']}_{row['week']}_{row['player_id']}",
        "season": int(row["season"]),
        "week": int(row["week"]),

        "player_info": {
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "team": row["team"]
        },

        "game_context": {
            "opponent": row["opponent"],
            "is_home": True
        },

        "pregame_form": {
            "yards_last3": row["yards_last3"],
            "yards_last5": row["yards_last5"],
            "tds_last3": row["tds_last3"],
            "atts_last3": row["atts_last3"],
            "yards_per_attempt": row["yards_per_attempt"]
        },

        "opponent_context": {
            "opp_pass_yards_pg": row["def_yards_pg"],
            "opp_pass_tds_pg": row["def_tds_pg"]
        },

        "targets": {
            "passing_yards": row["passing_yards"],
            "passing_tds": row["passing_tds"]
        }
    }
