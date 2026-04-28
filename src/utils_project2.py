from pathlib import Path
import logging
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd


# --------------------------------------------------
# setup folders and logging
# --------------------------------------------------

Path("logs").mkdir(parents=True, exist_ok=True)
Path("data/interim").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)


def setup_logger(log_file: str) -> None:
    """
    set up logging to a file.
    this should be called once near the start of each script.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True
    )
    logging.info("logging initialized")


# --------------------------------------------------
# conversion helpers
# --------------------------------------------------

def safe_to_pandas(df_like):
    """
    convert a polars dataframe or pandas dataframe into pandas.
    nflreadpy often returns polars, so this helper keeps the code flexible.
    """
    try:
        if hasattr(df_like, "to_pandas"):
            return df_like.to_pandas()
        return pd.DataFrame(df_like)
    except Exception as e:
        logging.exception("failed to convert object to pandas")
        raise e


# --------------------------------------------------
# column helpers
# --------------------------------------------------

def find_first_existing_column(df: pd.DataFrame, candidates: List[str], required: bool = True):
    """
    return the first column name from candidates that exists in df.
    this helps us write more defensive code if nflverse columns vary.
    """
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise KeyError(f"none of these columns were found: {candidates}")
    return None


# --------------------------------------------------
# feature engineering helpers
# --------------------------------------------------

def add_days_rest(df: pd.DataFrame, player_col: str, date_col: str) -> pd.DataFrame:
    """
    compute days since the player's previous game.
    this is a pregame feature and should be based on earlier rows only.
    """
    df = df.sort_values([player_col, date_col]).copy()
    df["days_rest"] = (
        df.groupby(player_col)[date_col]
        .diff()
        .dt.days
    )
    return df


def add_rolling_mean(
    df: pd.DataFrame,
    group_col: str,
    source_col: str,
    window: int,
    new_col: str
) -> pd.DataFrame:
    """
    create a rolling average using only previous games.
    shift(1) is critical so the current game is not included.
    """
    df = df.sort_values([group_col, "game_date"]).copy()

    df[new_col] = (
        df.groupby(group_col)[source_col]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    )
    return df


def add_cumulative_mean(
    df: pd.DataFrame,
    group_col: str,
    source_col: str,
    new_col: str
) -> pd.DataFrame:
    """
    create a season-to-date or career-to-date mean using only previous games.
    """
    df = df.sort_values([group_col, "game_date"]).copy()

    def _cummean_prior(s):
        prior_sum = s.shift(1).cumsum()
        prior_count = s.shift(1).expanding().count()
        return prior_sum / prior_count

    df[new_col] = df.groupby(group_col)[source_col].transform(_cummean_prior)
    return df


def add_prior_game_count(df: pd.DataFrame, group_col: str, new_col: str = "games_played_before") -> pd.DataFrame:
    """
    count how many prior games each player had before the current game.
    """
    df = df.sort_values([group_col, "game_date"]).copy()
    df[new_col] = df.groupby(group_col).cumcount()
    return df


# --------------------------------------------------
# opponent context helpers
# --------------------------------------------------

def build_opponent_allowed_features(
    qb_df: pd.DataFrame,
    opponent_col: str,
    date_col: str,
    yards_col: str,
    tds_col: str,
    attempts_col: str,
    completions_col: str
) -> pd.DataFrame:
    """
    for each opponent team, compute pass defense metrics allowed before the current game.
    since qb_df has one row per qb-game, grouping by opponent gives a rough defense-facing table.
    """
    temp = qb_df.sort_values([opponent_col, date_col]).copy()

    temp["opp_pass_yards_allowed_pg"] = (
        temp.groupby(opponent_col)[yards_col]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    temp["opp_pass_tds_allowed_pg"] = (
        temp.groupby(opponent_col)[tds_col]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    temp["opp_attempts_faced_pg"] = (
        temp.groupby(opponent_col)[attempts_col]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    temp["opp_completions_allowed_pg"] = (
        temp.groupby(opponent_col)[completions_col]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    return temp


# --------------------------------------------------
# serialization helpers
# --------------------------------------------------

def df_to_mongo_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    convert the final flat modeling dataframe into nested mongo-ready documents.
    """
    docs = []

    for _, row in df.iterrows():
        doc = {
            "_id": row["_id"],
            "game_id": row["game_id"],
            "season": int(row["season"]),
            "week": int(row["week"]),
            "game_date": str(row["game_date"].date()) if pd.notnull(row["game_date"]) else None,
            "game_type": row["game_type"],

            "player_info": {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "position": row["position"],
                "team": row["team"]
            },

            "game_context": {
                "team": row["team"],
                "opponent": row["opponent"],
                "is_home": bool(row["is_home"]) if pd.notnull(row["is_home"]) else None,
                "days_rest": float(row["days_rest"]) if pd.notnull(row["days_rest"]) else None
            },

            "pregame_form": {
                "games_played_before": int(row["games_played_before"]) if pd.notnull(row["games_played_before"]) else None,
                "avg_pass_yards_last_3": float(row["avg_pass_yards_last_3"]) if pd.notnull(row["avg_pass_yards_last_3"]) else None,
                "avg_pass_yards_last_5": float(row["avg_pass_yards_last_5"]) if pd.notnull(row["avg_pass_yards_last_5"]) else None,
                "avg_pass_tds_last_3": float(row["avg_pass_tds_last_3"]) if pd.notnull(row["avg_pass_tds_last_3"]) else None,
                "avg_pass_tds_last_5": float(row["avg_pass_tds_last_5"]) if pd.notnull(row["avg_pass_tds_last_5"]) else None,
                "avg_attempts_last_3": float(row["avg_attempts_last_3"]) if pd.notnull(row["avg_attempts_last_3"]) else None,
                "avg_attempts_last_5": float(row["avg_attempts_last_5"]) if pd.notnull(row["avg_attempts_last_5"]) else None,
                "avg_completions_last_3": float(row["avg_completions_last_3"]) if pd.notnull(row["avg_completions_last_3"]) else None,
                "avg_completions_last_5": float(row["avg_completions_last_5"]) if pd.notnull(row["avg_completions_last_5"]) else None,
                "avg_ints_last_3": float(row["avg_ints_last_3"]) if pd.notnull(row["avg_ints_last_3"]) else None,
                "season_to_date_pass_yards_pg": float(row["season_to_date_pass_yards_pg"]) if pd.notnull(row["season_to_date_pass_yards_pg"]) else None,
                "season_to_date_pass_tds_pg": float(row["season_to_date_pass_tds_pg"]) if pd.notnull(row["season_to_date_pass_tds_pg"]) else None,
                "season_to_date_attempts_pg": float(row["season_to_date_attempts_pg"]) if pd.notnull(row["season_to_date_attempts_pg"]) else None,
                "season_to_date_comp_pct": float(row["season_to_date_comp_pct"]) if pd.notnull(row["season_to_date_comp_pct"]) else None,
                "season_to_date_yards_per_attempt": float(row["season_to_date_yards_per_attempt"]) if pd.notnull(row["season_to_date_yards_per_attempt"]) else None
            },

            "opponent_context": {
                "opp_pass_yards_allowed_pg": float(row["opp_pass_yards_allowed_pg"]) if pd.notnull(row["opp_pass_yards_allowed_pg"]) else None,
                "opp_pass_tds_allowed_pg": float(row["opp_pass_tds_allowed_pg"]) if pd.notnull(row["opp_pass_tds_allowed_pg"]) else None,
                "opp_attempts_faced_pg": float(row["opp_attempts_faced_pg"]) if pd.notnull(row["opp_attempts_faced_pg"]) else None,
                "opp_completions_allowed_pg": float(row["opp_completions_allowed_pg"]) if pd.notnull(row["opp_completions_allowed_pg"]) else None
            },

            "targets": {
                "passing_yards": float(row["passing_yards"]),
                "passing_tds": float(row["passing_tds"])
            }
        }
        docs.append(doc)

    return docs


def write_json_documents(documents: List[Dict[str, Any]], output_path: str) -> None:
    """
    write a list of mongo-ready documents to a json file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)

    logging.info("wrote %s documents to %s", len(documents), output_path)


# --------------------------------------------------
# flattening helper for notebook use
# --------------------------------------------------

def flatten_mongo_docs(docs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    flatten nested mongo documents into a dataframe for modeling.
    """
    return pd.json_normalize(docs)
