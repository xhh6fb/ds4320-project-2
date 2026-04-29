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

### Item 3. Data Dictionary

| Name | Data type | Description | Example |
|---|---|---|---|
| `_id` | string | Custom unique identifier for one quarterback-game document | `2024_03_BUF_JAX_00-0034857` |
| `game_id` | string | Source game identifier when available | `2024_03_BUF_JAX` |
| `season` | integer | NFL season year | `2024` |
| `week` | integer | Regular-season week number | `3` |
| `game_date` | date/string | Calendar date of the game | `2024-09-23` |
| `game_type` | string | Type of game included in the dataset | `REG` |
| `player_info.player_id` | string | Quarterback player identifier | `00-0034857` |
| `player_info.player_name` | string | Quarterback name | `Josh Allen` |
| `player_info.position` | string | Player position | `QB` |
| `player_info.team` | string | Quarterback team abbreviation | `BUF` |
| `game_context.team` | string | Team abbreviation for the quarterback in that game | `BUF` |
| `game_context.opponent` | string | Opposing team abbreviation | `JAX` |
| `game_context.is_home` | boolean | Whether the quarterback’s team was home | `False` |
| `game_context.days_rest` | numeric | Days since the quarterback’s previous game | `7` |
| `pregame_form.games_played_before` | integer | Number of prior games before the current game | `2` |
| `pregame_form.avg_pass_yards_last_3` | numeric | Rolling average passing yards over the previous 3 games only | `254.5` |
| `pregame_form.avg_pass_yards_last_5` | numeric | Rolling average passing yards over the previous 5 games only | `267.8` |
| `pregame_form.avg_pass_tds_last_3` | numeric | Rolling average passing touchdowns over the previous 3 games only | `2.0` |
| `pregame_form.avg_pass_tds_last_5` | numeric | Rolling average passing touchdowns over the previous 5 games only | `1.8` |
| `pregame_form.avg_attempts_last_3` | numeric | Rolling average pass attempts over the previous 3 games only | `35.7` |
| `pregame_form.avg_attempts_last_5` | numeric | Rolling average pass attempts over the previous 5 games only | `34.4` |
| `pregame_form.avg_completions_last_3` | numeric | Rolling average completions over the previous 3 games only | `24.3` |
| `pregame_form.avg_completions_last_5` | numeric | Rolling average completions over the previous 5 games only | `23.8` |
| `pregame_form.avg_ints_last_3` | numeric | Rolling average interceptions over the previous 3 games only | `0.7` |
| `pregame_form.season_to_date_pass_yards_pg` | numeric | Prior season-to-date passing yards per game | `262.1` |
| `pregame_form.season_to_date_pass_tds_pg` | numeric | Prior season-to-date passing touchdowns per game | `1.9` |
| `pregame_form.season_to_date_attempts_pg` | numeric | Prior season-to-date pass attempts per game | `34.6` |
| `pregame_form.season_to_date_comp_pct` | numeric | Prior season-to-date completion percentage | `0.678` |
| `pregame_form.season_to_date_yards_per_attempt` | numeric | Prior season-to-date yards per attempt | `7.57` |
| `opponent_context.opp_pass_yards_allowed_pg` | numeric | Opponent’s pregame average passing yards allowed per game | `226.0` |
| `opponent_context.opp_pass_tds_allowed_pg` | numeric | Opponent’s pregame average passing TDs allowed per game | `1.5` |
| `opponent_context.opp_attempts_faced_pg` | numeric | Opponent’s pregame average pass attempts faced per game | `33.1` |
| `opponent_context.opp_completions_allowed_pg` | numeric | Opponent’s pregame average completions allowed per game | `22.9` |
| `targets.passing_yards` | numeric | Actual passing yards in the game; prediction target | `263` |
| `targets.passing_tds` | numeric | Actual passing touchdowns in the game; prediction target | `2` |

### Quantification of Uncertainty for Numerical Features - FIX!

To quantify uncertainty in the numerical features, I summarize the observed distributions of the main engineered variables in the final dataset. Rather than describing uncertainty only in words, I report numerical distribution summaries including the mean, standard deviation, minimum, quartiles, maximum, and missingness rate. This helps show both the range of plausible values and where certain features may be sparse or highly variable.

| Feature | Mean | Std Dev | Min | 25th pct | Median | 75th pct | Max | Missing % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pregame_form.avg_pass_yards_last_3` | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE |
| `pregame_form.avg_pass_tds_last_3` | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE |
| `game_context.days_rest` | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE |
| `opponent_context.opp_pass_yards_allowed_pg` | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE |
| `opponent_context.opp_pass_tds_allowed_pg` | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE |
| `targets.passing_yards` | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE |
| `targets.passing_tds` | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE | REPLACE |
