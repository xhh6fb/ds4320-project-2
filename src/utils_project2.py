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

    logger.info("Adding rest days feature")

    # ensure datetime
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    df = df.sort_values(["player_id", "game_date"])

    df["prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)

    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days

    return df

# ============================================================
# WEATHER
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

    df["is_home"] = (df["team"] == df["home_team"]).astype(int)

    return df

# ============================================================
# QB LAST 5 GAMES AVERAGE YARDS (QB "SKILL")
# ============================================================

def add_qb_form(df, logger):

    logger.info("Adding QB last 5 games form")

    df = df.sort_values(["player_id", "game_date"])

    df["yards_last5"] = (
        df.groupby("player_id")["passing_yards"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )

    df["yards_last5_std"] = (
        df.groupby("player_id")["passing_yards"]
        .transform(lambda x: x.shift().rolling(5, min_periods=2).std())
    )

    return df

# ============================================================
# DEFENSE FEATURES
# ============================================================

def add_defense_features(pbp, logger):
    """
    Computes opponent defensive strength from the QB's perspective.
    All metrics are lagged (shift(1)) before rolling, so there is
    no data leakage from the current game.
 
    Metrics computed per game for each defensive team, then rolled:
        opp_epa_per_pass_allowed   — avg EPA allowed per pass attempt
        opp_success_rate_allowed   — share of pass plays with EPA > 0
        opp_air_yards_allowed      — avg air yards allowed per attempt
        opp_yac_allowed            — avg yards after catch allowed per attempt
 
    FIX vs prior version:
        - Filters to pass plays only before aggregating
        - Sorts by (season, week) — not just week — so cross-season
          ordering is correct
        - Drops intermediate columns before returning
    """
 
    logger.info("Adding defensive strength features")
 
    # filter to pass plays only
    passes = pbp[pbp["pass_attempt"] == 1].copy()
 
    # one row per (season, week, defteam) game 
    defense = passes.groupby(
        ["season", "week", "defteam"]
    ).agg(
        _epa = ("epa", "mean"),
        _success_rate = ("success", "mean"),
        _air_yards = ("air_yards", "mean"),
        _yac = ("yards_after_catch", "mean"),
    ).reset_index()
 
    # sort correctly across seasons
    defense = defense.sort_values(["defteam", "season", "week"])
 
    # rolling 5-game lagged average (no leakage)
    rolling_map = {
        "_epa": "opp_epa_per_pass_allowed",
        "_success_rate": "opp_success_rate_allowed",
        "_air_yards": "opp_air_yards_allowed",
        "_yac": "opp_yac_allowed",
    }
 
    for raw_col, feature_col in rolling_map.items():
        defense[feature_col] = (
            defense.groupby("defteam")[raw_col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
 
    # drop raw intermediate columns
    defense.drop(columns=list(rolling_map.keys()), inplace=True)
 
    # rename for merge key
    defense.rename(columns={"defteam": "opponent"}, inplace=True)
 
    return defense
