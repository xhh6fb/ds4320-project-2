import pandas as pd
import logging


def setup_logger(log_file_name):
    logger = logging.getLogger(log_file_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file_name)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def add_rest_days(df, logger):
    logger.info("Adding rest days")

    df = df.sort_values(["player_id", "season", "week"])
    df["prev_game"] = df.groupby("player_id")["game_date"].shift(1)

    df["days_rest"] = (df["game_date"] - df["prev_game"]).dt.days

    return df


# -----------------------------
# CORE ROLLING FEATURES
# -----------------------------
def add_rolling_features(df, logger):
    logger.info("Rolling features (3-game)")

    df = df.sort_values(["player_id", "season", "week"])

    df["yards_last3"] = (
        df.groupby("player_id")["passing_yards"]
        .transform(lambda x: x.shift().rolling(3).mean())
    )

    df["tds_last3"] = (
        df.groupby("player_id")["passing_tds"]
        .transform(lambda x: x.shift().rolling(3).mean())
    )

    df["atts_last3"] = (
        df.groupby("player_id")["attempts"]
        .transform(lambda x: x.shift().rolling(3).mean())
    )

    return df


# -----------------------------
# EXTRA FEATURES FOR YOUR MODEL
# -----------------------------
def add_extended_rolling_features(df, logger):
    logger.info("Extended rolling features")

    df["yards_last5"] = (
        df.groupby("player_id")["passing_yards"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )

    df["yards_per_attempt"] = df["passing_yards"] / df["attempts"]

    return df


# -----------------------------
# DEFENSE FEATURES
# -----------------------------
def add_defense_features(pbp, logger):
    logger.info("Defense features")

    defense = pbp.groupby(["season", "week", "defteam"]).agg(
        def_pass_yards=("passing_yards", "sum"),
        def_pass_tds=("pass_touchdown", "sum"),
        def_pass_att=("pass_attempt", "sum")
    ).reset_index()

    defense = defense.sort_values(["defteam", "season", "week"])

    defense["def_yards_pg"] = (
        defense.groupby("defteam")["def_pass_yards"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )

    defense["def_tds_pg"] = (
        defense.groupby("defteam")["def_pass_tds"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )

    defense.rename(columns={"defteam": "team"}, inplace=True)

    return defense


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
