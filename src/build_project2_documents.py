from pathlib import Path
import logging

import numpy as np
import pandas as pd
import nflreadpy as nfl

from utils_project2 import (
    setup_logger,
    safe_to_pandas,
    find_first_existing_column,
    add_days_rest,
    add_rolling_mean,
    add_cumulative_mean,
    add_prior_game_count,
    build_opponent_allowed_features,
    df_to_mongo_documents,
    write_json_documents
)


def main():
    """
    build a custom qb-game document dataset for predicting next-game
    passing yards and passing touchdowns using only pregame information.
    """
    setup_logger("logs/build_project2_documents.log")
    logging.info("starting build_project2_documents.py")

    try:
        Path("data/interim").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # --------------------------------------------------
        # 1. load raw nflverse data through nflreadpy
        # --------------------------------------------------

        seasons = list(range(2016, 2026))

        logging.info("loading player stats")
        player_stats_raw = safe_to_pandas(nfl.load_player_stats(seasons=seasons))

        logging.info("loading schedules")
        schedules_raw = safe_to_pandas(nfl.load_schedules(seasons=seasons))

        logging.info("loading players")
        players_raw = safe_to_pandas(nfl.load_players())

        logging.info("player_stats_raw shape: %s", player_stats_raw.shape)
        logging.info("schedules_raw shape: %s", schedules_raw.shape)
        logging.info("players_raw shape: %s", players_raw.shape)

        # save raw-ish copies for transparency/debugging
        player_stats_raw.to_csv("data/interim/player_stats_raw.csv", index=False)
        schedules_raw.to_csv("data/interim/schedules_raw.csv", index=False)

        # --------------------------------------------------
        # 2. inspect and standardize expected columns
        # --------------------------------------------------

        # because public sports data can evolve, we defensively search for likely column names
        player_id_col = find_first_existing_column(player_stats_raw, ["player_id", "gsis_id"])
        player_name_col = find_first_existing_column(player_stats_raw, ["player_name", "player_display_name", "name"])
        season_col = find_first_existing_column(player_stats_raw, ["season"])
        week_col = find_first_existing_column(player_stats_raw, ["week"])
        recent_team_col = find_first_existing_column(player_stats_raw, ["recent_team", "team", "team_abbr"])
        position_col = find_first_existing_column(player_stats_raw, ["position", "position_group"], required=False)
        game_type_col = find_first_existing_column(player_stats_raw, ["season_type", "game_type"], required=False)

        passing_yards_col = find_first_existing_column(player_stats_raw, ["passing_yards", "pass_yards"])
        passing_tds_col = find_first_existing_column(player_stats_raw, ["passing_tds", "pass_tds"])
        attempts_col = find_first_existing_column(player_stats_raw, ["attempts", "pass_attempts", "passing_attempts"])
        completions_col = find_first_existing_column(player_stats_raw, ["completions", "passing_completions", "pass_completions"])
        ints_col = find_first_existing_column(player_stats_raw, ["interceptions", "passing_interceptions", "pass_interceptions"])

        # schedule columns
        sched_season_col = find_first_existing_column(schedules_raw, ["season"])
        sched_week_col = find_first_existing_column(schedules_raw, ["week"])
        home_team_col = find_first_existing_column(schedules_raw, ["home_team"])
        away_team_col = find_first_existing_column(schedules_raw, ["away_team"])
        gameday_col = find_first_existing_column(schedules_raw, ["gameday", "game_date"])
        sched_game_type_col = find_first_existing_column(schedules_raw, ["game_type", "season_type"], required=False)
        game_id_col = find_first_existing_column(schedules_raw, ["game_id", "old_game_id"], required=False)

        # players table columns
        players_id_col = find_first_existing_column(players_raw, ["player_id", "gsis_id"])
        players_name_col = find_first_existing_column(players_raw, ["display_name", "player_name", "full_name", "short_name"], required=False)
        players_pos_col = find_first_existing_column(players_raw, ["position", "position_group"], required=False)

        # --------------------------------------------------
        # 3. select only fields we need and clean
        # --------------------------------------------------

        qb = player_stats_raw[[
            player_id_col,
            player_name_col,
            season_col,
            week_col,
            recent_team_col,
            passing_yards_col,
            passing_tds_col,
            attempts_col,
            completions_col,
            ints_col
        ]].copy()

        qb.columns = [
            "player_id",
            "player_name",
            "season",
            "week",
            "team",
            "passing_yards",
            "passing_tds",
            "attempts",
            "completions",
            "interceptions"
        ]

        # add position if available in player_stats, otherwise merge from players
        if position_col is not None:
            qb["position"] = player_stats_raw[position_col].values
        else:
            qb["position"] = np.nan

        # add game type if available in player_stats
        if game_type_col is not None:
            qb["game_type"] = player_stats_raw[game_type_col].values
        else:
            qb["game_type"] = np.nan

        # merge with players table to fill missing name/position info
        players_small = players_raw[[players_id_col]].copy()
        players_small.columns = ["player_id"]

        if players_name_col is not None:
            players_small["player_name_master"] = players_raw[players_name_col].values

        if players_pos_col is not None:
            players_small["position_master"] = players_raw[players_pos_col].values

        qb = qb.merge(players_small.drop_duplicates("player_id"), on="player_id", how="left")

        qb["player_name"] = qb["player_name"].fillna(qb.get("player_name_master"))
        qb["position"] = qb["position"].fillna(qb.get("position_master"))

        # keep only qbs
        qb = qb[qb["position"].astype(str).str.upper() == "QB"].copy()

        # keep only regular season if game type exists
        if "game_type" in qb.columns and qb["game_type"].notna().any():
            qb = qb[qb["game_type"].astype(str).str.upper().isin(["REG", "R", "REGULAR"])].copy()
            qb["game_type"] = "REG"
        else:
            # if player stats game type is missing, infer from schedules later
            qb["game_type"] = np.nan

        # keep only rows with meaningful passing activity
        qb = qb[qb["attempts"].fillna(0) >= 5].copy()

        # --------------------------------------------------
        # 4. clean schedules and merge to get opponent / home-away / date
        # --------------------------------------------------

        schedules = schedules_raw[[
            sched_season_col,
            sched_week_col,
            home_team_col,
            away_team_col,
            gameday_col
        ]].copy()

        schedules.columns = ["season", "week", "home_team", "away_team", "game_date"]

        if sched_game_type_col is not None:
            schedules["game_type"] = schedules_raw[sched_game_type_col].values
        else:
            schedules["game_type"] = np.nan

        if game_id_col is not None:
            schedules["game_id"] = schedules_raw[game_id_col].values
        else:
            schedules["game_id"] = np.nan

        schedules["game_date"] = pd.to_datetime(schedules["game_date"], errors="coerce")

        # regular season only
        if schedules["game_type"].notna().any():
            schedules = schedules[schedules["game_type"].astype(str).str.upper().isin(["REG", "R", "REGULAR"])].copy()

        # create team-perspective schedule rows so we can merge by qb team
        sched_home = schedules.copy()
        sched_home["team"] = sched_home["home_team"]
        sched_home["opponent"] = sched_home["away_team"]
        sched_home["is_home"] = True

        sched_away = schedules.copy()
        sched_away["team"] = sched_away["away_team"]
        sched_away["opponent"] = sched_away["home_team"]
        sched_away["is_home"] = False

        sched_long = pd.concat([sched_home, sched_away], ignore_index=True)

        qb = qb.merge(
            sched_long[["season", "week", "team", "opponent", "is_home", "game_date", "game_id"]],
            on=["season", "week", "team"],
            how="left"
        )

        # if player_stats did not carry game type consistently, assign reg after merge
        qb["game_type"] = "REG"

        # remove rows that still failed to merge to a game date/opponent
        qb = qb[qb["game_date"].notna()].copy()

        # --------------------------------------------------
        # 5. sort and build pregame features
        # --------------------------------------------------

        qb = qb.sort_values(["player_id", "game_date"]).reset_index(drop=True)

        # prior game count
        qb = add_prior_game_count(qb, "player_id", "games_played_before")

        # rest days
        qb = add_days_rest(qb, "player_id", "game_date")

        # rolling features
        qb = add_rolling_mean(qb, "player_id", "passing_yards", 3, "avg_pass_yards_last_3")
        qb = add_rolling_mean(qb, "player_id", "passing_yards", 5, "avg_pass_yards_last_5")

        qb = add_rolling_mean(qb, "player_id", "passing_tds", 3, "avg_pass_tds_last_3")
        qb = add_rolling_mean(qb, "player_id", "passing_tds", 5, "avg_pass_tds_last_5")

        qb = add_rolling_mean(qb, "player_id", "attempts", 3, "avg_attempts_last_3")
        qb = add_rolling_mean(qb, "player_id", "attempts", 5, "avg_attempts_last_5")

        qb = add_rolling_mean(qb, "player_id", "completions", 3, "avg_completions_last_3")
        qb = add_rolling_mean(qb, "player_id", "completions", 5, "avg_completions_last_5")

        qb = add_rolling_mean(qb, "player_id", "interceptions", 3, "avg_ints_last_3")

        # season-to-date style averages based only on prior games
        qb = add_cumulative_mean(qb, "player_id", "passing_yards", "season_to_date_pass_yards_pg")
        qb = add_cumulative_mean(qb, "player_id", "passing_tds", "season_to_date_pass_tds_pg")
        qb = add_cumulative_mean(qb, "player_id", "attempts", "season_to_date_attempts_pg")

        # comp pct and yards per attempt
        qb["comp_pct"] = np.where(qb["attempts"] > 0, qb["completions"] / qb["attempts"], np.nan)
        qb["yards_per_attempt"] = np.where(qb["attempts"] > 0, qb["passing_yards"] / qb["attempts"], np.nan)

        qb = add_cumulative_mean(qb, "player_id", "comp_pct", "season_to_date_comp_pct")
        qb = add_cumulative_mean(qb, "player_id", "yards_per_attempt", "season_to_date_yards_per_attempt")

        # opponent defensive context
        qb = build_opponent_allowed_features(
            qb_df=qb,
            opponent_col="opponent",
            date_col="game_date",
            yards_col="passing_yards",
            tds_col="passing_tds",
            attempts_col="attempts",
            completions_col="completions"
        )

        # --------------------------------------------------
        # 6. final cleaning for project dataset
        # --------------------------------------------------

        # create a stable custom id for each document
        qb["_id"] = (
            qb["season"].astype(str) + "_" +
            qb["week"].astype(str).str.zfill(2) + "_" +
            qb["team"].astype(str) + "_" +
            qb["opponent"].astype(str) + "_" +
            qb["player_id"].astype(str)
        )

        # keep only columns needed in final dataset
        final_cols = [
            "_id",
            "game_id",
            "season",
            "week",
            "game_date",
            "game_type",
            "player_id",
            "player_name",
            "position",
            "team",
            "opponent",
            "is_home",
            "days_rest",
            "games_played_before",
            "avg_pass_yards_last_3",
            "avg_pass_yards_last_5",
            "avg_pass_tds_last_3",
            "avg_pass_tds_last_5",
            "avg_attempts_last_3",
            "avg_attempts_last_5",
            "avg_completions_last_3",
            "avg_completions_last_5",
            "avg_ints_last_3",
            "season_to_date_pass_yards_pg",
            "season_to_date_pass_tds_pg",
            "season_to_date_attempts_pg",
            "season_to_date_comp_pct",
            "season_to_date_yards_per_attempt",
            "opp_pass_yards_allowed_pg",
            "opp_pass_tds_allowed_pg",
            "opp_attempts_faced_pg",
            "opp_completions_allowed_pg",
            "passing_yards",
            "passing_tds"
        ]

        final_df = qb[final_cols].copy()

        # optional: remove the very first career game for each qb because many pregame fields are empty
        final_df = final_df[final_df["games_played_before"] >= 1].copy()

        # save a flat custom dataset for transparency
        final_df.to_csv("data/processed/qb_model_flat.csv", index=False)
        logging.info("saved flat custom dataset: %s rows", len(final_df))

        # create mongo-ready nested documents
        mongo_docs = df_to_mongo_documents(final_df)
        write_json_documents(mongo_docs, "data/processed/qb_game_documents.json")

        logging.info("finished build successfully")
        print("done: built custom qb dataset and mongo json documents")
        print(f"flat rows: {len(final_df):,}")
        print(f"documents: {len(mongo_docs):,}")
        print("saved to data/processed/qb_model_flat.csv")
        print("saved to data/processed/qb_game_documents.json")

    except Exception as e:
        logging.exception("build failed")
        raise e


if __name__ == "__main__":
    main()
