# DS 4320 Project 2: Predicting NFL QB Passing Production

## Executive Summary

This repository contains my DS 4320 project on predicting NFL quarterback next-game passing production. I built a custom secondary dataset from public NFL source data accessed through the `nflreadpy` Python package, transformed it into nested MongoDB documents, and stored it in MongoDB Atlas. Each document represents one quarterback-game observation and includes player identity, game context, opponent context, and rolling pregame passing features tailored to the problem of projecting next-game passing yards and passing touchdowns using only information available before kickoff. The repository contains the Python scripts used to build and load the data, a Jupyter notebook analysis pipeline and markdown export, a separate press release, metadata describing the document structure and feature definitions, and background readings to help explain the project domain.

<br>

||Details|
|---|---|
| Name | Jolie Ng |
| NetID | xhh6fb |
| DOI | [![DOI - CHANGE!](https://zenodo.org/badge/DOI/10.5281/zenodo.19363443.svg)](https://doi.org/10.5281/zenodo.19363443) |
| Press Release | [NFL Document Database Uses Pregame Trends to Project QB Performance](press_release.md) |
| Pipeline | [Notebook - UPLOAD!](pipeline/project_2_pipeline.ipynb) & [Markdown - UPLOAD!](pipeline/project_2_pipeline.md)   |
| License | [MIT](LICENSE) |

<br>

## Problem Definition

### Initial General Problem & Refined Specific Problem Statement

**Initial General Problem:** Projecting athletic performance.

**Refined Specific Problem Statement:** Build a document-model NFL dataset in MongoDB using nflverse data and use it to project next-game passing performance for NFL quarterbacks, with a specific focus on predicting pregame passing yards and touchdown production from information available before kickoff.

### Motivation

Projecting athletic performance is one of the most practically useful problems in sports analytics because teams, media, fantasy players, and analysts all want to estimate what a player is likely to do next. In the NFL, quarterback performance is especially important because it strongly influences team offense, game flow, and final outcomes. I chose this project because it combines my interest in football analytics with a realistic modeling problem that can be supported by public data. It also fits the document model well, since player-week records can naturally be stored as rich nested documents that combine player identity, team context, opponent context, rolling performance trends, and game metadata in one place.

### Rationale for Refinement

I refined the problem from broad athletic performance to quarterback next-game passing performance because the original problem is too large and ambiguous. Athletic performance could refer to many sports, many player roles, and many target variables. Quarterbacks offer a narrower and more coherent modeling target because their performance is measured every week with widely used statistics such as passing yards, passing touchdowns, interceptions, completions, and attempts. Restricting the project to pregame prediction also prevents leakage by ensuring that only information available before the game is used as input features. This refinement makes the project more analytically sound, easier to evaluate, and more suitable for a semester-scale document database project.

### Press Release

[NFL Document Database Uses Pregame Trends to Project QB Performance](press_release.md)

<br>

## Domain Exposition

### Terminology

| Term | Meaning | Why It Matters |
|---|---|---|
| NFL | National Football League | Add text |
| QB | Quarterback | Add text |
| Passing yards | Total yards gained through completed forward passes | Add text |
| Passing touchdowns | Number of touchdown passes thrown | Add text |
| Interceptions | Passes thrown to the defense | Add text |
| Completion percentage | Completions divided by pass attempts | Add text |
| Attempt volume | Number of passes thrown in a game | Add text |
| Pregame feature | A variable known before kickoff | Add text |
| Rolling average | An average calculated from prior games only | Add text |
| Opponent defense | The defense the quarterback will face in the upcoming game | Add text |
| Home/away split | Whether the game is played at the QB’s home stadium or away | Add text |
| Rest days | Number of days since the player’s previous game | Add text |
| Document model | A database model that stores data as nested documents, such as JSON-like objects | Add text |
| MongoDB | A document-oriented database system | Add text |
| nflverse | An open NFL analytics data ecosystem | Add text |
| nflreadpy | The Python package used to access nflverse data in Python | Add text |

### Domain

This project lives in the domain of sports analytics, specifically NFL player performance modeling. Sports analytics uses historical game and player data to answer questions about prediction, evaluation, and strategy. Within that domain, quarterback performance is one of the most studied topics because passing production often drives offensive success. The project uses public NFL data to construct player-week documents that combine historical performance, recent form, opponent context, and scheduling context into a structure suitable for modeling. The goal is not just to describe what happened, but to use pregame information to estimate what is likely to happen next.

### Background Reading

The [`background_reading`](background_reading) folder contains readings that help explain the football analytics context of this project.

| Index | Title | Brief Description | Path |
|---|---|---|---|
| 1 | nflWAR: A Reproducible Method for Offensive Player Evaluation in Football | Reproducible NFL offensive player evaluation using WAR models | [Link](background_reading/01_nflwar.pdf) |
| 2 | A Reinforcement Learning Based Approach to Play Calling in Football | Uses reinforcement learning to optimize football play calling decisions | [Link](background_reading/02_reinforcement_learning_approach_to_play_calling.pdf) |
| 3 | NFL Play Prediction | Machine learning models predict NFL play outcomes and yardage | [Link](background_reading/03_nfl_play_prediction.pdf) |
| 4 | The Quarterback Prediction Problem: Forecasting NFL QB Performance | Shows difficulty predicting QB performance from pre-draft data | [Link](background_reading/04_quarterback_prediction_problem.pdf) |
| 5 | next-gen-scraPy: Extracting NFL Tracking Data from Images to Evaluate Quarterbacks and Pass Defenses | Extracts tracking data to evaluate quarterbacks and pass defenses | [Link](background_reading/05_next_gen_scrapy.pdf) |

<br>

## Data Creation

### Provenance

The raw data for this project comes from the nflverse ecosystem, which is a public collection of NFL analytics data and tools. In Python, I plan to access the data through the `nflreadpy` package, which is the maintained Python interface for loading nflverse datasets. The project will primarily use schedule, team, and player-stat data that can be joined conceptually by season, week, team, game, and player identifiers. The goal is to create a secondary dataset focused on projecting quarterback passing performance before each game.

The raw acquisition process begins by downloading the relevant nflverse tables into Python dataframes. I will then filter the data to the seasons and game types I want to study, most likely regular-season NFL games only, because that gives a more consistent context than mixing preseason and postseason games together. After that, I will construct player-game records and then transform them into document-style records for MongoDB. Each final document will represent a quarterback-game observation with nested subfields for player identity, team context, opponent context, schedule context, and rolling pregame statistics. This means the final project dataset is a secondary dataset built from public raw NFL source data instead of being manually collected firsthand observations.

### Data Creation Code - FIX!

The code below shows the python file used to create the secondary dataset and support the project pipeline.

| File | Description |
|---|---|
| `build_project2_documents.py` | Loads raw nflverse data in Python, filters seasons/games, computes rolling pregame quarterback features, and writes JSON-like documents for MongoDB import |
| `load_project2_to_mongo.py` | Connects to MongoDB Atlas and inserts the prepared quarterback documents into the target collection |
| `project2_pipeline.ipynb` | Queries MongoDB into dataframes, performs the analysis/modeling pipeline, and creates visualizations |
| `project2_pipeline.md` | Markdown export of the notebook pipeline |
| `utils_project2.py` | Helper functions for logging, validation, rolling feature creation, and safe transformations |

| File | What It Does | Path |
|---|---|---|
| `pipeline/build_project_tables.py` | Loads raw data from `nflreadpy`, filters to completed regular-season games, creates the `teams`, `games`, `team_games`, and `matchups` tables, loads them into DuckDB, and exports the final tables as parquet files | [Link](pipeline/build_project_tables.py) |

### Bias Identification

Bias can enter the data collection process in several ways. First, the project only includes games and statistics that are publicly available in nflverse, so it is limited to what those source systems track and publish. Second, if I restrict the analysis to regular-season games, quarterbacks with unusual postseason performance or limited sample sizes may be represented differently than long-term starters. Third, quarterback opportunity itself is not random: players on stronger offenses, with better pass protection, more aggressive coaches, or healthier receiving groups may accumulate more favorable statistics. As a result, the dataset may reflect organizational and usage differences in addition to pure individual ability.

### Bias Mitigation

I can mitigate these biases by being explicit about the scope of the data, modeling only clearly defined populations, and reporting uncertainty rather than overclaiming precision. For example, I can restrict the problem to quarterbacks with a minimum number of attempts or starts so the target population is more comparable. I can also include contextual features such as home/away status, opponent, rest days, and recent usage volume so the model captures some environmental effects instead of attributing everything to the player alone. In reporting results, I should compare errors across player groups and note that the model estimates expected performance within the observed historical environment, not a pure measure of talent.

### Rationale for Critical Decisions

Several critical decisions shape the dataset and introduce uncertainty. One major decision is to focus on quarterbacks rather than all player positions. I made that choice because quarterbacks have more consistent weekly passing statistics and a clearer modeling target. Another key decision is to use only pregame information when constructing features. That is important because including current-game information would create leakage and make the model unrealistically strong. I also need to decide how to summarize prior performance, such as whether to use season-to-date averages, rolling windows, or exponential weighting. Different summary choices can materially change the signal available to the model.

There are also judgment calls around which games to include. Restricting the dataset to regular-season games improves comparability, but it also excludes playoff environments. Handling missing data is another source of uncertainty: some players may have incomplete histories, injuries, or very small sample sizes. Finally, in a document-model dataset, I must decide how much information to embed in each quarterback-game document. Embedding too little makes the documents weak for downstream analysis, while embedding too much can duplicate information and complicate updates. I will favor documents that are rich enough for analysis but still logically organized into substructures like `player_info`, `team_context`, `opponent_context`, and `pregame_form`.

<br>

## Metadata - FIX!

### Implicit Schema - FIX!

Since this is a document-model project, the logical schema is a soft schema** instead of a fully normalized relational schema. The core collection will be a quarterback-game collection in MongoDB.

**quarterbacks_games (collection structure)**

| Field |
|---|
| _id |
| game_id |
| season |
| week |
| gameday |
| player_info { ... } |
| team_context { ... } |
| opponent_context { ... } |
| pregame_form { ... } |
| outcome { ... } |

**Logical structure of one document**

#### player_info
- player_id  
- player_name  
- team  
- position  

#### team_context
- is_home  
- days_rest  
- team_record_before  
- team_points_for_pg_before  
- team_points_against_pg_before  

#### opponent_context
- opponent_team  
- opponent_record_before  
- opponent_points_allowed_pg_before  

#### pregame_form
- games_played_before  
- pass_attempts_pg_before  
- completions_pg_before  
- completion_pct_before  
- pass_yards_pg_before  
- pass_tds_pg_before  
- interceptions_pg_before  

#### outcome
- actual_pass_yards  
- actual_pass_tds  
- actual_interceptions  

### Data Summary - FIX!

Rather than using a raw online table directly as the final dataset, I created a custom secondary dataset specifically tailored to the project question of predicting quarterback next-game passing yards and passing touchdowns. The raw source data came from nflverse through the `nflreadpy` Python package, but I filtered those raw tables to regular-season quarterback observations, merged schedule context, identified the quarterback’s opponent and home/away status, calculated rest days, and engineered rolling pregame performance summaries. I also created opponent pass-defense context features based on what opposing defenses had allowed before each game. The result is a purpose-built quarterback-game document dataset designed for pregame forecasting rather than a generic download of public source data.

| Collection | Description |
|---|---|
| Data Source | nflverse (via nflreadpy Python package) |
| Database | MongoDB Atlas |
| Database Name | nfl_project |
| quarterbacks_games.json | Main document dataset; one document per quarterback-game observation with nested pregame context and game outcome fields |
| quarterbacks_games_sample.csv | Flat CSV export of a subset of analysis-ready fields for inspection |

| Data | Description | Path |
|---|---|---|
| `teams` | Team reference table containing identifiers and metadata | [`teams.parquet`](https://myuva-my.sharepoint.com/:u:/g/personal/xhh6fb_virginia_edu/IQCs78KkRH5fRZO4i7-LF2KQAb_8JcCiCFyla5uDszr-xcI?e=Ioi9xR) |
| `games` | One row per completed NFL regular-season game with schedule and market context | [`games.parquet`](https://myuva-my.sharepoint.com/:u:/g/personal/xhh6fb_virginia_edu/IQD3pgX0GCf_Sqk9_HAyxO1eASPFUFW0U3NLJZ33NGCv6fA?e=PqbryA) |
| `team_games` | One row per team per game with leakage-safe rolling pregame features | [`team_games.parquet`](https://myuva-my.sharepoint.com/:u:/g/personal/xhh6fb_virginia_edu/IQAeTonPHtfSSqj6gHfRQEhFAdWSt3PmzrVopXmSZppRGY0?e=Hq4mPA) |
| `matchups` | One row per game containing final home/away pregame features for modeling | [`matchups.parquet`](https://myuva-my.sharepoint.com/:u:/g/personal/xhh6fb_virginia_edu/IQDPH5QAzl6ASoSPrYcyS_KtAYzLKyagYTTsdMfJyXCBw_k?e=GsMW2H) |

### Data Dictionary - FIX!

This table defines each field in the MongoDB document structure.

| Name | Data type | Description | Example |
|---|---|---|---|
| game_id | string | Unique game identifier | 2023_10_DAL_PHI |
| season | integer | NFL season year | 2023 |
| week | integer | Week of the season | 10 |
| gameday | date/string | Date of the game | 2023-11-05 |
| player_info.player_id | string | Unique quarterback/player identifier | 00-0036355 |
| player_info.player_name | string | Quarterback name | Jalen Hurts |
| player_info.team | string | Player team abbreviation | PHI |
| player_info.position | string | Position label | QB |
| team_context.is_home | boolean | Whether the team is home | true |
| team_context.days_rest | integer | Days since previous game | 7 |
| team_context.team_record_before | string | Team record before kickoff | 8-1 |
| team_context.team_points_for_pg_before | float | Avg points scored per game before kickoff | 28.4 |
| team_context.team_points_against_pg_before | float | Avg points allowed per game before kickoff | 21.1 |
| opponent_context.opponent_team | string | Opponent abbreviation | DAL |
| opponent_context.opponent_record_before | string | Opponent record before kickoff | 5-3 |
| opponent_context.opponent_points_allowed_pg_before | float | Opponent points allowed per game | 19.8 |
| pregame_form.games_played_before | integer | Prior games used for features | 8 |
| pregame_form.pass_attempts_pg_before | float | Avg pass attempts before game | 34.5 |
| pregame_form.completions_pg_before | float | Avg completions before game | 23.9 |
| pregame_form.completion_pct_before | float | Completion rate before game | 0.693 |
| pregame_form.pass_yards_pg_before | float | Avg passing yards before game | 258.7 |
| pregame_form.pass_tds_pg_before | float | Avg passing TDs before game | 1.9 |
| pregame_form.interceptions_pg_before | float | Avg interceptions before game | 0.8 |
| outcome.actual_pass_yards | integer | Passing yards in game | 279 |
| outcome.actual_pass_tds | integer | Passing TDs in game | 2 |
| outcome.actual_interceptions | integer | Interceptions in game | 1 |

#### `teams`

| Feature | Data type | Description | Example |
|---|---|---|---|
| `team_id` | string | Team abbreviation used as key | `BAL` |
| `team_name` | string | Full team name | `Baltimore Ravens` |
| `team_nick` | string | Team nickname | `Ravens` |
| `team_conf` | string | Conference | `AFC` |
| `team_division` | string | Division | `AFC North` |

#### `games`

| Feature | Data type | Description | Example |
|---|---|---|---|
| `game_id` | string | Unique game identifier | `2023_W01_HOU_at_BAL` |
| `season` | integer | NFL season year | `2023` |
| `week` | integer | Regular-season week number | `1` |
| `gameday` | date | Date of game | `2023-09-10` |
| `home_team` | string | Home team ID | `BAL` |
| `away_team` | string | Away team ID | `HOU` |
| `home_score` | integer | Home final score | `25` |
| `away_score` | integer | Away final score | `9` |
| `home_rest` | numeric | Pregame home-team rest days from source data when available | `7` |
| `away_rest` | numeric | Pregame away-team rest days from source data when available | `7` |
| `home_moneyline` | numeric | Pregame home-team American moneyline | `-450` |
| `away_moneyline` | numeric | Pregame away-team American moneyline | `350` |
| `spread_line` | numeric | Pregame point spread | `9.5` |
| `total_line` | numeric | Pregame projected total points | `43.5` |
| `div_game` | integer/boolean | Indicator for division matchup | `0` |
| `roof` | string | Stadium roof type when available | `outdoor` |
| `surface` | string | Field surface when available | `grass` |
| `temp` | numeric | Pregame temperature when available | `75` |
| `wind` | numeric | Pregame wind speed when available | `8` |
| `home_win` | integer | 1 if home team won, else 0 | `1` |
| `winner_team` | string | Winner team ID | `BAL` |
| `market_home_implied_prob` | float | Implied home-team win probability from home moneyline | `0.818` |
| `market_away_implied_prob` | float | Implied away-team win probability from away moneyline | `0.222` |
| `market_implied_prob_diff` | float | Home implied probability minus away implied probability | `0.596` |
| `sched_rest_diff` | numeric | Home rest minus away rest | `0` |

#### `team_games`

| Feature | Data type | Description | Example |
|---|---|---|---|
| `game_id` | string | Foreign key to game | `2023_W01_HOU_at_BAL` |
| `team_id` | string | Team for the row | `BAL` |
| `opponent_team` | string | Opponent team ID | `HOU` |
| `season` | integer | Season year | `2023` |
| `week` | integer | Week number | `1` |
| `gameday` | date | Game date | `2023-09-10` |
| `is_home` | integer | 1 if home, 0 if away | `1` |
| `points_for` | integer | Points scored by the team | `25` |
| `points_against` | integer | Points allowed by the team | `9` |
| `win` | integer | 1 if team won, else 0 | `1` |
| `games_played_before` | integer | Number of prior games entering this game | `0` |
| `cum_wins_before` | integer | Prior cumulative wins | `0` |
| `cum_losses_before` | integer | Prior cumulative losses | `0` |
| `cum_points_for_before` | numeric | Prior cumulative points scored | `0` |
| `cum_points_against_before` | numeric | Prior cumulative points allowed | `0` |
| `pregame_win_pct` | float | Win percentage before this game | `0.500` |
| `pregame_points_for_pg` | float | Prior scoring average | `0.0` |
| `pregame_points_against_pg` | float | Prior allowed average | `0.0` |
| `pregame_point_diff_pg` | float | Prior point differential average | `0.0` |
| `pregame_last3_points_for_pg` | float | Average points scored over previous 3 games | `0.0` |
| `pregame_last3_points_against_pg` | float | Average points allowed over previous 3 games | `0.0` |
| `pregame_last3_win_pct` | float | Win percentage over previous 3 games | `0.500` |
| `pregame_last3_point_diff_pg` | float | Recent-form point differential over previous 3 games | `0.0` |
| `days_rest_calc` | numeric | Calculated days since prior game | `7` |

#### `matchups`

| Feature | Data type | Description | Example |
|---|---|---|---|
| `game_id` | string | Unique game identifier | `2023_W01_HOU_at_BAL` |
| `season` | integer | Season year | `2023` |
| `week` | integer | Week number | `1` |
| `gameday` | date | Game day | `2023-09-10` |
| `home_team` | string | Home team ID | `BAL` |
| `away_team` | string | Away team ID | `HOU` |
| `home_score` | integer | Home final score | `25` |
| `away_score` | integer | Away final score | `9` |
| `home_rest` | numeric | Pregame home-team rest from source data | `7` |
| `away_rest` | numeric | Pregame away-team rest from source data | `7` |
| `home_moneyline` | numeric | Pregame home moneyline | `-450` |
| `away_moneyline` | numeric | Pregame away moneyline | `350` |
| `spread_line` | numeric | Pregame spread | `9.5` |
| `total_line` | numeric | Pregame projected total points | `43.5` |
| `div_game` | integer/boolean | Division-game indicator | `0` |
| `roof` | string | Roof type | `outdoor` |
| `surface` | string | Field surface | `grass` |
| `temp` | numeric | Pregame temperature | `75` |
| `wind` | numeric | Pregame wind speed | `8` |
| `home_win` | integer | 1 if home team won, else 0 | `1` |
| `winner_team` | string | Winner team ID | `BAL` |
| `market_home_implied_prob` | float | Implied home win probability from moneyline | `0.818` |
| `market_away_implied_prob` | float | Implied away win probability from moneyline | `0.222` |
| `market_implied_prob_diff` | float | Home implied probability minus away implied probability | `0.596` |
| `sched_rest_diff` | numeric | Home source rest minus away source rest | `0` |
| `home_games_played_before` | integer | Prior home-team games played | `0` |
| `away_games_played_before` | integer | Prior away-team games played | `0` |
| `home_cum_wins_before` | integer | Prior cumulative home-team wins | `0` |
| `away_cum_wins_before` | integer | Prior cumulative away-team wins | `0` |
| `home_cum_losses_before` | integer | Prior cumulative home-team losses | `0` |
| `away_cum_losses_before` | integer | Prior cumulative away-team losses | `0` |
| `home_pregame_win_pct` | float | Home-team pregame win percentage | `0.500` |
| `away_pregame_win_pct` | float | Away-team pregame win percentage | `0.500` |
| `home_pregame_points_for_pg` | float | Home-team prior scoring average | `0.0` |
| `away_pregame_points_for_pg` | float | Away-team prior scoring average | `0.0` |
| `home_pregame_points_against_pg` | float | Home-team prior points allowed average | `0.0` |
| `away_pregame_points_against_pg` | float | Away-team prior points allowed average | `0.0` |
| `home_pregame_point_diff_pg` | float | Home-team prior point differential average | `0.0` |
| `away_pregame_point_diff_pg` | float | Away-team prior point differential average | `0.0` |
| `home_pregame_last3_win_pct` | float | Home-team recent 3-game win percentage | `0.500` |
| `away_pregame_last3_win_pct` | float | Away-team recent 3-game win percentage | `0.500` |
| `home_pregame_last3_point_diff_pg` | float | Home-team recent 3-game point differential average | `0.0` |
| `away_pregame_last3_point_diff_pg` | float | Away-team recent 3-game point differential average | `0.0` |
| `home_days_rest_calc` | numeric | Calculated home-team rest days | `7` |
| `away_days_rest_calc` | numeric | Calculated away-team rest days | `7` |
| `pregame_win_pct_diff` | float | Home minus away pregame win percentage | `0.0` |
| `pregame_points_for_pg_diff` | float | Home minus away prior scoring average | `0.0` |
| `pregame_points_against_pg_diff` | float | Home minus away prior allowed average | `0.0` |
| `pregame_point_diff_pg_diff` | float | Home minus away prior point differential average | `0.0` |
| `games_played_before_diff` | integer | Home minus away prior games played | `0` |
| `cum_wins_before_diff` | integer | Home minus away cumulative wins | `0` |
| `cum_losses_before_diff` | integer | Home minus away cumulative losses | `0` |
| `last3_win_pct_diff` | float | Home minus away recent 3-game win percentage | `0.0` |
| `last3_point_diff_pg_diff` | float | Home minus away recent 3-game point differential average | `0.0` |
| `calc_rest_diff` | numeric | Calculated home rest minus away rest | `0` |
| `target_home_win` | integer | Final prediction target | `1` |

### Quantification of Uncertainty for Numerical Features - FIX!

Numerical features in this dataset contain inherent uncertainty mainly due to variability in player performance and external game conditions.

| Numerical feature | Uncertainty / Limitation |
|---|---|
| team_context.days_rest | Usually exact if schedule data is complete, but can be affected by bye weeks, international games, or missing prior game links |
| team_context.team_points_for_pg_before | Depends on prior games; early-season values are unstable due to small sample sizes |
| team_context.team_points_against_pg_before | Similar instability as points-for; sensitive to opponent strength and early-season variance |
| opponent_context.opponent_points_allowed_pg_before | Sensitive to small sample sizes and changes in defensive strength over time |
| pregame_form.games_played_before | Exact if data is complete, but may vary depending on inclusion of partial appearances |
| pregame_form.pass_attempts_pg_before | Unstable for quarterbacks with few starts or inconsistent playing time |
| pregame_form.completions_pg_before | Depends heavily on sample size and game participation consistency |
| pregame_form.completion_pct_before | Highly sensitive to small denominators and low-attempt games |
| pregame_form.pass_yards_pg_before | High variance early in season; influenced by matchup and game script |
| pregame_form.pass_tds_pg_before | Sparse event → high noise and instability |
| pregame_form.interceptions_pg_before | Rare event → highly volatile and situation-dependent |
| outcome.actual_pass_yards | Observed statistic; may be updated if official stats are corrected |
| outcome.actual_pass_tds | Typically exact once games are finalized |
| outcome.actual_interceptions | Typically exact once official scoring is finalized |

**Quantification methods**

To account for uncertainty, I plan on using the following approaches:

- Variance and standard deviation of rolling features
- Error residuals between predicted and actual outcomes
- Comparison of different rolling window sizes (e.g. 3 vs 5 games)
- Model evaluation using RMSE / MAE (if extended to modeling stage)

**Summary**

Overall, we can't remove uncertainty, but we can explicitly acknowledge and measure it. This is consistent with real-world sports prediction systems where outcomes are inherently stochastic.

| Numerical feature | Source of uncertainty | Quantitative discussion |
|---|---|---|
| `home_score / away_score` | Officially recorded outcomes | These are observed final values and are not used as predictors in the final model because they occur after kickoff |
| `pregame_win_pct` | Small-sample instability early in the season | When `games_played_before` is small, one game can change the estimate sharply; after 1 prior game the value can only be 0 or 1 |
| `pregame_points_for_pg` | Sampling variation across prior games | Early-season averages are noisy because one unusually high- or low-scoring game strongly changes the estimate |
| `pregame_points_against_pg` | Sampling variation across prior games | Defensive estimates also stabilize only after several earlier games |
| `pregame_point_diff_pg` | Inherits uncertainty from two averages | Since it is built from prior points scored and allowed, it compounds uncertainty from both |
| `pregame_last3_win_pct` | Short-window volatility | This captures recent form but is intentionally sensitive to short-term variation, especially when fewer than 3 prior games exist |
| `pregame_last3_point_diff_pg` | Short-window volatility | A three-game rolling average is responsive but noisier than full-history averages |
| `home_rest / away_rest` | Source-field incompleteness | These values can be missing in some rows and may not capture all fatigue-related context such as travel burden |
| `market_home_implied_prob / market_away_implied_prob` | Derived from moneylines | These are deterministic transformations of betting odds, but betting lines themselves reflect market opinion rather than certainty |
| `spread_line / total_line` | Pregame market estimate | These are strong predictors, but they represent public expectations and can still be wrong on individual games |
| `temp / wind` | Environmental context | These can help provide context but may be missing or imperfectly reflect playing conditions inside specific stadium settings |
| `difference features` | Propagated uncertainty from both teams | Each home-minus-away feature combines uncertainty from two separate team-level estimates, especially in early weeks |
