import pandas as pd
import logging

# ================================
# LOGGER SETUP
# ================================
def setup_logger(log_file_name):
    """
    Creates a logger that writes pipeline logs to a file.
    Useful for debugging ETL + feature pipeline issues.
    """

    logger = logging.getLogger(log_file_name)

    # prevent duplicate logs if re-run in notebook
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file_name)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def add_rest_days(df, logger):
    logger.info("Adding rest days")

    df = df.sort_values(["player_id", "season", "week"])
    df["prev_game"] = df.groupby("player_id")["game_date"].shift(1)

    df["days_rest"] = (df["game_date"] - df["prev_game"]).dt.days

    return df


# ================================
# ROLLING QB FORM FEATURES
# ================================
def add_rolling_features(df, logger):
    """
    Creates short-term QB performance features.
    IMPORTANT: uses shift() to prevent data leakage.
    """

    logger.info("Adding rolling QB features (3-game window)")

    df = df.sort_values(["player_id", "game_date"])

    # last 3 games passing yards
    df["yards_last3"] = (
        df.groupby("player_id")["passing_yards"]
        .transform(lambda x: x.shift().rolling(3).mean())
    )

    # last 5 games passing yards
    df["yards_last5"] = (
        df.groupby("player_id")["passing_yards"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )

    # last 3 games TDs
    df["tds_last3"] = (
        df.groupby("player_id")["passing_tds"]
        .transform(lambda x: x.shift().rolling(3).mean())
    )

    # last 3 games attempts
    df["atts_last3"] = (
        df.groupby("player_id")["attempts"]
        .transform(lambda x: x.shift().rolling(3).mean())
    )

    return df


# ================================
# EXTENDED EFFICIENCY FEATURES
# ================================
def add_extended_rolling_features(df, logger):
    """
    Adds derived efficiency metrics.
    """

    logger.info("Adding extended QB efficiency features")

    # efficiency per attempt
    df["yards_per_attempt"] = df["passing_yards"] / df["attempts"]

    return df


# ================================
# DEFENSE STRENGTH FEATURES
# ================================
def add_defense_features(pbp, qb, logger):
    """
    Builds opponent defensive strength metrics
    and merges them into QB dataset.
    """

    logger.info("Building defensive strength features")

    # ================================
    # AGGREGATE DEFENSE PERFORMANCE
    # ================================
    defense = pbp.groupby(
        ["season", "week", "defteam"]
    ).agg(
        def_pass_yards=("passing_yards", "sum"),
        def_pass_tds=("pass_touchdown", "sum"),
        def_pass_attempts=("pass_attempt", "sum")
    ).reset_index()

    # rename for merging
    defense.rename(columns={"defteam": "opponent"}, inplace=True)

    # ================================
    # SORT FOR TIME SERIES LOGIC
    # ================================
    defense = defense.sort_values(["opponent", "season", "week"])

    # ================================
    # ROLLING DEFENSE STRENGTH (5-week window)
    # ================================
    defense["def_yards_pg"] = (
        defense.groupby("opponent")["def_pass_yards"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )

    defense["def_tds_pg"] = (
        defense.groupby("opponent")["def_pass_tds"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )

    # ================================
    # MERGE INTO QB DATASET
    # ================================
    qb = qb.merge(
        defense,
        on=["season", "week", "opponent"],
        how="left"
    )

    return qb


# -----------------------------
# MONGO DOCUMENT FORMAT
# -----------------------------
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
            "opponent": row["opponent"]
        },

        "pregame_form": {
            "yards_last3": row["yards_last3"],
            "yards_last5": row["yards_last5"],
            "tds_last3": row["tds_last3"],
            "atts_last3": row["atts_last3"],
            "yards_per_attempt": row["yards_per_attempt"]
        },

        "opponent_context": {
            "opp_def_yards_pg": row["def_yards_pg"],
            "opp_def_tds_pg": row["def_tds_pg"]
        },

        "targets": {
            "passing_yards": row["passing_yards"],
            "passing_tds": row["passing_tds"]
        }
    }

