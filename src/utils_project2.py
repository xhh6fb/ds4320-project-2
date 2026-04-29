import pandas as pd
import logging

# ============================================================
# LOGGER SETUP
# ============================================================
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

# ============================================================
# REST DAYS (PREVIOUS GAME ONLY SO NO LEAKAGE)
# ============================================================
def add_rest_days(df, logger):

    logger.info("Adding rest days")

    df = df.sort_values(["passer_id", "season", "week"])

    df["prev_game_date"] = df.groupby("passer_id")["game_date"].shift(1)

    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days

    return df

# ============================================================
# WEATHER CLEANING
# ============================================================
def add_weather_features(df, logger):

    logger.info("Processing weather features")

    # convert to numeric safely
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce")

    df["bad_weather"] = ((df["wind"] > 15) | (df["temp"] < 32)).astype(int)

    return df

# ============================================================
# HOME OR AWAY
# ============================================================
def add_home_away(df, logger):

    logger.info("Adding home/away feature")

    df["is_home"] = (df["posteam"] == df["home_team"]).astype(int)

    return df

# ============================================================
# QB LAST 5 GAMES AVERAGE YARDS (QB "SKILL")
# ============================================================
def add_qb_form(df, logger):

    logger.info("Adding QB last 5 games form")

    df = df.sort_values(["passer_id", "game_date"])

    df["yards_last5"] = (
        df.groupby("passer_id")["passing_yards"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )

    return df

# ============================================================
# DEFENSE LAST 5 GAMES AVERAGE YARDS (DEFENSE "SKILL")
# ============================================================
def add_defense_features(df, logger):

    logger.info("Adding defensive strength features")

    defense = df.groupby(
        ["season", "week", "defteam"]
    ).agg(
        def_pass_yards=("passing_yards", "mean"),
    ).reset_index()

    defense = defense.sort_values(["defteam", "season", "week"])

    defense["def_yards_pg"] = (
        defense.groupby("defteam")["def_pass_yards"]
        .transform(lambda x: x.shift().rolling(5).mean())
    )

    defense.rename(columns={"defteam": "opponent"}, inplace=True)

    return defense
