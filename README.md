# DS 4320 Project 2: Predicting NFL QB Passing Production

## Executive Summary
 
This repository contains my DS 4320 project on predicting NFL quarterback next-game passing production using only structural, contextual, and opponent-facing features — deliberately excluding a quarterback's own historical passing yards as a direct predictor. I built a custom secondary dataset from public NFL source data accessed through the `nfl_data_py` Python package, transformed it into nested MongoDB documents, and stored it in MongoDB Atlas. Each document represents one quarterback-game observation and includes player identity, game context (including betting market lines), opponent defensive quality metrics derived from EPA and advanced pass-defense statistics, weather, usage share, and rolling pregame features — all constructed to contain zero information from the current game. The research question is whether structural information about game environment, opponent defense, and usage role can predict passing volume on its own — without ever feeding the model how many yards that quarterback threw for last week. The repository contains the Python scripts used to build and load the data, a Jupyter notebook analysis pipeline and markdown export, a separate press release, metadata describing the document structure and feature definitions, and background readings explaining the project domain.

<br>

||Details|
|---|---|
| Name | Jolie Ng |
| NetID | xhh6fb |
| DOI | [![DOI - CHANGE!](https://zenodo.org/badge/DOI/10.5281/zenodo.19363443.svg)](https://doi.org/10.5281/zenodo.19363443) |
| Press Release | [Can Context Alone Predict a Quarterback's Passing Day?](press_release.md) |
| Pipeline | [Notebook - UPLOAD!](pipeline/project_2_pipeline.ipynb) & [Markdown - UPLOAD!](pipeline/project_2_pipeline.md)   |
| License | [MIT](LICENSE) |

<br>

## Problem Definition
 
### Initial General Problem & Refined Specific Problem Statement
 
**Initial General Problem:** Projecting athletic performance.
 
**Refined Specific Problem Statement:** Build a document-model NFL dataset in MongoDB using nflverse data and use it to project next-game passing yards for NFL quarterbacks, specifically testing whether structural pregame features (betting market lines, opponent pass-defense quality, weather, usage share, and scheduling context) can produce accurate predictions *without* relying on the quarterback's own historical passing yard totals as input features.

### Motivation
 
Quarterback performance is arguably the single most consequential variable in determining NFL game outcomes, and accurately projecting it in advance has enormous value across domains including team analytics, sports media, and fantasy sports. Most prediction approaches lean heavily on a quarterback's recent statistical history, treating last week's yards as the strongest signal for next week's. But this creates a practical blind spot: a newly installed starter, a quarterback returning from injury, or a player who just switched teams may have little or no usable history. More fundamentally, raw yard totals conflate many underlying factors — opponent difficulty, game script, team pass volume, and weather — that could be modeled directly from information available before kickoff. This project tests whether those structural factors, captured through tools like EPA-based opponent defense ratings, betting market total lines, and each quarterback's share of their team's pass volume, contain enough signal on their own to project passing production meaningfully.
 
### Rationale for Refinement
 
I refined the problem from broad athletic performance prediction to a targeted test of whether contextual features alone can predict quarterback passing yards. The general problem is too wide: it spans multiple sports, positions, and target variables, and the most obvious modeling approach — feed in recent stats, get a prediction — answers a different, much easier question. The interesting and harder question is what happens when you take the historical yard totals away. If you can still predict passing production reasonably well from environment alone, that tells you something important: game context and opponent quality are doing most of the explanatory work, not the quarterback's individual talent signal contained in recent yardage. That is a more analytically useful finding, and it produces a model that is robust precisely in the situations where history-based models break down — new starters, injury returns, team changes, and early-season observations with limited prior data.
 
### Press Release
 
[Can Context Alone Predict a Quarterback's Passing Day?](press_release.md)
 
<br>

## Domain Exposition
 
### Terminology

| Term | Meaning | Why It Matters |
|---|---|---|
| NFL | National Football League | The data-generating environment for the entire project. All observations, features, and predictions are based on NFL game and player data. |
| QB | Quarterback | The focus of this project; quarterbacks produce consistent weekly passing statistics, making them the natural unit of observation for a game-by-game prediction problem. |
| Passing yards | Total yards gained through completed forward passes | The primary prediction target for this project. |
| Pass attempt | A forward pass thrown by the quarterback, regardless of outcome | The main volume measure; more attempts generally means more yards, making attempt share a key structural predictor. |
| EPA | Expected Points Added | A play-level measure of how much a play improves or harms a team's expected scoring. Used here to measure opponent pass-defense quality. |
| Success rate | Share of pass plays where EPA > 0 | An opponent-defense quality metric measuring how often the defense shuts down passing plays, regardless of yards. |
| Air yards | Yards the ball travels through the air from the line of scrimmage to where it is caught or incomplete | A pregame opponent metric indicating how much downfield the defense allows the passing game to operate. |
| Yards after catch (YAC) | Yards a receiver gains after the ball is caught | A pregame opponent metric indicating the defense's ability to limit yards after contact. |
| QB pass share | Player pass attempts divided by total team pass attempts in the same game | A usage metric capturing how much of the team's passing volume runs through the starting quarterback. |
| Spread line | The point spread set by oddsmakers for the game | A market-based pregame signal for expected game script; large spreads imply one team will throw more to catch up. |
| Total line | The over/under points total set by oddsmakers | A market-based signal for expected offensive volume in the game. |
| Pregame feature | Any variable known before kickoff | All input features in this model must be pregame features to ensure there is no data leakage from the current game. |
| Rolling average | An average calculated from prior games only, with a lag applied | Used to summarize opponent pass-defense trends without including current-game information. |
| Leakage | Using current-game information as a model input | A methodological error that makes the model unrealistically accurate in training but useless in real prediction. |
| Game script | The in-game strategic tendency driven by score and game state | Trailing teams throw more; market lines capture pre-game expectations of game script. |
| Home/away split | Whether a game is played at the QB's home stadium or away | Crowd noise, travel fatigue, and scheduling factors can affect performance; included as a binary indicator. |
| Rest days | Days since the player's previous game | Short rest (Thursday games) can affect performance; included as a scheduling feature. |
| Bad weather | Binary flag for wind > 15 mph or temperature < 32°F | Extreme weather suppresses passing volume and makes the passing game harder to execute. |
| Document model | A database structure that stores records as nested JSON-like documents | Used to store each quarterback-game observation as a self-contained document with nested context subfields. |
| MongoDB | A document-oriented database system | The storage system used for the custom quarterback-game dataset. |
| nflverse | An open NFL analytics data ecosystem | The source of all raw data used to build the custom secondary dataset. |
| nfl_data_py | The Python package used to access nflverse play-by-play data | The tool used to load raw NFL play-by-play tables into Python for cleaning, aggregation, and feature engineering. |
 
### Domain
 
This project lives in the domain of sports analytics, with a specific focus on NFL offensive performance modeling. Sports analytics uses historical game and player data to answer questions about evaluation, prediction, and strategy. Within that domain, quarterback passing projection is one of the most studied problems because passing production directly drives scoring and game outcomes. This project takes a non-standard approach by restricting the model's inputs to structural, contextual features — opponent pass-defense quality, betting market lines, usage share, weather, and scheduling — and excluding the quarterback's own historical passing yardage. The domain-specific insight motivating this design is that game context explains a large share of passing volume variance: a quarterback facing a weak defense in a high-total game with expected negative game script will throw more, regardless of what they did last week. Testing whether that hypothesis holds quantitatively requires deliberately removing the convenience of historical yards and seeing what remains.
 
### Background Reading
 
The [`background_reading`](background_reading) folder contains readings explaining the football analytics context of this project.

| Index | Title | Brief Description | Path |
|---|---|---|---|
| 1 | nflWAR: A Reproducible Method for Offensive Player Evaluation in Football | Reproducible NFL offensive player evaluation using WAR models | [Link](background_reading/01_nflwar.pdf) |
| 2 | A Reinforcement Learning Based Approach to Play Calling in Football | Uses reinforcement learning to optimize football play-calling decisions | [Link](background_reading/02_reinforcement_learning_approach_to_play_calling.pdf) |
| 3 | NFL Play Prediction | Machine learning models predicting NFL play outcomes and yardage | [Link](background_reading/03_nfl_play_prediction.pdf) |
| 4 | The Quarterback Prediction Problem: Forecasting NFL QB Performance | Illustrates the difficulty of predicting QB performance from limited or pre-draft data | [Link](background_reading/04_quarterback_prediction_problem.pdf) |
| 5 | next-gen-scraPy: Extracting NFL Tracking Data to Evaluate Quarterbacks and Pass Defenses | Extracts tracking data to evaluate quarterbacks and pass defenses using air yards and route metrics | [Link](background_reading/05_next_gen_scrapy.pdf) |

<br>

## Data Creation
 
### Provenance
 
The raw data for this project comes from the nflverse ecosystem, a public collection of NFL analytics data and tooling. In Python, I access the data through the `nfl_data_py` package, which is the maintained Python interface for loading nflverse datasets. The primary raw input is play-by-play data covering the 2020 through 2024 NFL regular seasons, loaded via `nfl.import_pbp_data()`. Each row in the play-by-play table represents one play and includes the passer, teams, outcome, EPA, air yards, yards after catch, weather conditions, betting lines, and dozens of other contextual fields.
 
From this raw table, I construct two types of aggregations. The first is a quarterback-game aggregation that produces one row per quarterback per game, summing passing yards and attempts and taking the first observed value for game-level constants like weather and market lines. The second is a defensive aggregation that computes, for each team-game on defense, the average EPA per pass attempt, success rate, air yards allowed, and YAC allowed — then rolls those metrics over the prior five games with a one-game lag so that no current-game information enters the features. The two aggregations are then joined on season, week, and opponent to produce the final analysis-ready dataset. The final dataset is a secondary, purpose-built dataset created from public raw source data rather than directly downloaded or manually collected.
 
### Data Creation Code
 
| File | Description | Path |
|---|---|---|
| `src/build_project2.py` | Loads raw nflverse play-by-play data, aggregates to QB-game level, computes team-level pass attempts for usage share, merges betting line and weather features, calls the utils feature pipeline, joins defensive context, and writes the final CSV | [`src/build_project2.py`](src/build_project2.py) |
| `src/utils_project2.py` | Helper functions for logging, QB rolling form, rest days, weather flags, home/away indicator, and the rolling opponent pass-defense feature pipeline | [`src/utils_project2.py`](src/utils_project2.py) |
| `src/load_project2_to_mongo.py` | Connects to MongoDB Atlas and inserts prepared quarterback documents into the target collection | [`src/load_project2_to_mongo.py`](src/load_project2_to_mongo.py) |
| `src/sample_mongosh_queries.js` | Example `mongosh` commands for checking, querying, and summarizing the MongoDB collection | [`src/sample_mongosh_queries.js`](src/sample_mongosh_queries.js) |
| `pipeline/project2_pipeline.ipynb` | Queries MongoDB into a dataframe, runs the modeling pipeline excluding historical yards features, evaluates predictions, and generates visualizations | [`pipeline/project2_pipeline.ipynb`](pipeline/project2_pipeline.ipynb) |
| `pipeline/project2_pipeline.md` | Markdown export of the notebook pipeline | [`pipeline/project2_pipeline.md`](pipeline/project2_pipeline.md) |
 
### Bias Identification
 
Bias can enter the data collection and feature engineering process in several ways. First, the dataset only includes game observations where a quarterback threw enough passes to be identified as the primary passer, which means backup quarterbacks or game managers with very low attempt volumes may be underrepresented or excluded by the `dropna()` filter applied after the rolling window calculations. Second, quarterbacks on strong offensive teams are likely to have higher `qb_pass_share` values by default, meaning the usage feature implicitly reflects team-level pass-heavy tendencies rather than purely individual workload. Third, the rolling opponent defense metrics — EPA allowed, success rate, air yards, and YAC — reflect the schedule of opponents each defense has faced, so a defense that plays weak offenses early in the season will look stronger than it is. Fourth, by excluding historical passing yards as a predictor, the model may perform systematically worse for quarterbacks whose passing volume is unusually stable or unusually volatile, since those patterns cannot be captured without the autoregressive signal.
 
### Bias Mitigation
 
I mitigate these biases through several design choices. The rolling opponent features use a five-game lag window with a one-game shift, which smooths out any single-game anomaly and prevents current-game leakage. The `min_periods=1` rolling parameter prevents early-season rows from being dropped purely due to insufficient history. Including betting market lines — spread and total — as features partially corrects for opponent schedule bias, since market makers implicitly account for strength of schedule when setting lines. To address the exclusion of backup quarterbacks, I restrict analysis to observations that survive the full feature pipeline without missingness, which in practice focuses the dataset on quarterbacks with meaningful sample sizes. In reporting model results, I compare errors across quarterback usage tiers and note explicitly that predictions for quarterbacks with limited prior observations in the dataset should be interpreted with additional uncertainty.
 
### Rationale for Critical Decisions
 
The most consequential decision in this project is the exclusion of historical passing yards as a direct predictor. This was a deliberate research design choice rather than a data availability constraint. Most naive models for projecting passing yards use `yards_last_n` as their strongest feature, but this conflates the question of predicting performance with the question of extrapolating autocorrelated time series. Excluding it forces the model to use features that explain *why* a quarterback might have a high-volume game — game script expectations from the spread, offense-defense matchup quality from EPA metrics, and usage role from pass share — rather than simply regressing on what happened before. This produces a model that generalizes better to situations without a useful history and generates more interpretable insights about what contextual drivers matter.
 
A second major decision is using play-by-play data rather than pre-aggregated box score data. This allows the defensive features to be computed correctly at the play level — filtering to pass attempts only before aggregating EPA, air yards, and YAC — which is more accurate than applying those metrics to all play types. A third decision is to sort defensive rolling features by `(defteam, season, week)` rather than just `(defteam, week)`, which is necessary to ensure correct temporal ordering across season boundaries. An earlier version of the code sorted only by week and would have incorrectly sequenced observations where week 1 of a new season followed week 18 of the prior season in the wrong order.
 
<br>

## Metadata
 
### Implicit Schema
 
The MongoDB collection stores one document per quarterback-game observation. Each document represents the quarterback's full pregame situation for a specific game, along with the actual game outcome used for prediction evaluation. The structure follows a consistent nested schema across all documents.
 
#### Top-level document fields
 
- `_id`: custom unique identifier combining game_id and player_id
- `game_id`: source game identifier (e.g., `2024_03_BUF_JAX`)
- `season`: NFL season year
- `week`: NFL week number
- `game_date`: calendar date of the game
#### Nested object: `player_info`
 
Stores quarterback identity.
 
- `player_id`: player identifier string
- `player_name`: quarterback display name
- `team`: quarterback's team abbreviation for that game
#### Nested object: `game_context`
 
Stores game-level contextual information available before kickoff.
 
- `opponent`: opposing team abbreviation
- `home_team`: home team abbreviation
- `away_team`: away team abbreviation
- `is_home`: binary indicator (1 = home, 0 = away)
- `days_rest`: days since the quarterback's previous game
- `temp`: temperature in degrees Fahrenheit at game time
- `wind`: wind speed in mph at game time
- `bad_weather`: binary flag (1 = wind > 15 mph or temp < 32°F)
- `time_of_day`: timestamp or time-of-day string for kickoff
- `spread_line`: point spread set by oddsmakers (negative favors the home team)
- `total_line`: over/under points total set by oddsmakers
#### Nested object: `usage_context`
 
Stores pregame usage and volume features.
 
- `pass_attempts`: quarterback's total pass attempts in the game (also used as part of target context)
- `team_attempts`: total team pass attempts in the game
- `qb_pass_share`: quarterback's pass attempts divided by team attempts
#### Nested object: `pregame_form`
 
Stores rolling quarterback trend features based on prior games only. These are present in the dataset but intentionally excluded as model predictors to test the structural-features-only hypothesis.
 
- `yards_last5`: rolling 5-game average passing yards (lagged — does not include current game)
- `yards_last5_std`: rolling 5-game standard deviation of passing yards (lagged)
#### Nested object: `opponent_context`
 
Stores opponent pass-defense quality features, all lagged and rolled over prior games.
 
- `opp_epa_per_pass_allowed`: rolling 5-game mean EPA per pass allowed by the opponent defense
- `opp_success_rate_allowed`: rolling 5-game mean pass-play success rate allowed
- `opp_air_yards_allowed`: rolling 5-game mean air yards per pass attempt allowed
- `opp_yac_allowed`: rolling 5-game mean yards after catch per attempt allowed
#### Nested object: `targets`
 
Stores actual postgame outcomes used for model evaluation.
 
- `passing_yards`: total passing yards in the game
#### Example document
 
```json
{
  "_id": "2024_03_NE_NYJ_00-0023459",
  "game_id": "2024_03_NE_NYJ",
  "season": 2024,
  "week": 3,
  "game_date": "2024-09-19",
  "player_info": {
    "player_id": "00-0023459",
    "player_name": "A.Rodgers",
    "team": "NYJ"
  },
  "game_context": {
    "opponent": "NE",
    "home_team": "NYJ",
    "away_team": "NE",
    "is_home": 1,
    "days_rest": 4,
    "temp": 75.0,
    "wind": 5.0,
    "bad_weather": 0,
    "time_of_day": "2024-09-20T00:17:35Z",
    "spread_line": 6.5,
    "total_line": 39.0
  },
  "usage_context": {
    "pass_attempts": 37,
    "team_attempts": 37,
    "qb_pass_share": 1.0
  },
  "pregame_form": {
    "yards_last5": 141.4,
    "yards_last5_std": 80.9
  },
  "opponent_context": {
    "opp_epa_per_pass_allowed": -0.180,
    "opp_success_rate_allowed": 0.395,
    "opp_air_yards_allowed": 8.34,
    "opp_yac_allowed": 4.88
  },
  "targets": {
    "passing_yards": 281
  }
}
```
 
### Data Summary
 
| Property | Value |
|---|---|
| Data Source | nflverse (via `nfl_data_py` Python package) |
| Seasons Covered | 2020–2024 NFL regular seasons and playoffs |
| Unit of Observation | One quarterback-game |
| Database | MongoDB Atlas |
| Database Name | `nfl_project` |
| Collection | `quarterback_games` |
| Main Dataset File | `data/qb_games.csv` |
| Primary Target Variable | `passing_yards` |
| Key Predictor Categories | Opponent defense (EPA, success rate, air yards, YAC), market lines (spread, total), weather, usage (qb_pass_share), scheduling (days_rest, is_home) |
| Excluded from Prediction | QB's own historical passing yards (`yards_last5`) |
 
### Data Dictionary
 
| Name | Data Type | Description | Example |
|---|---|---|---|
| `season` | integer | NFL season year | `2024` |
| `week` | integer | NFL week number within the season | `3` |
| `game_id` | string | Source game identifier | `2024_03_NE_NYJ` |
| `player_id` | string | Quarterback player identifier | `00-0023459` |
| `player_name` | string | Quarterback display name | `A.Rodgers` |
| `team` | string | Quarterback's team abbreviation for the game | `NYJ` |
| `opponent` | string | Opposing team abbreviation | `NE` |
| `home_team` | string | Home team abbreviation | `NYJ` |
| `away_team` | string | Away team abbreviation | `NE` |
| `game_date` | date | Calendar date of the game | `2024-09-19` |
| `passing_yards` | numeric | Total passing yards in the game — **prediction target** | `281` |
| `pass_attempts` | numeric | Total pass attempts by this quarterback in the game | `37` |
| `temp` | numeric | Temperature in degrees Fahrenheit at game time | `75.0` |
| `wind` | numeric | Wind speed in mph at game time | `5.0` |
| `time_of_day` | string | Kickoff timestamp | `2024-09-20T00:17:35Z` |
| `spread_line` | numeric | Pregame point spread (negative favors home team) | `6.5` |
| `total_line` | numeric | Pregame over/under points total | `39.0` |
| `team_attempts` | numeric | Total pass attempts by the QB's team in the game | `37` |
| `qb_pass_share` | numeric | QB's pass attempts divided by team attempts; measures usage share | `1.0` |
| `yards_last5` | numeric | Rolling 5-game average passing yards from prior games only; **excluded as model predictor** | `141.4` |
| `yards_last5_std` | numeric | Rolling 5-game standard deviation of passing yards from prior games only; **excluded as model predictor** | `80.9` |
| `prev_game_date` | date | Date of the quarterback's previous game | `2024-09-15` |
| `days_rest` | numeric | Days between the previous game and the current game | `4` |
| `bad_weather` | integer | Binary flag: 1 if wind > 15 mph or temp < 32°F, else 0 | `0` |
| `is_home` | integer | Binary flag: 1 if quarterback's team is the home team, else 0 | `1` |
| `opp_epa_per_pass_allowed` | numeric | Rolling 5-game lagged mean EPA per pass attempt allowed by the opponent defense | `-0.180` |
| `opp_success_rate_allowed` | numeric | Rolling 5-game lagged mean rate of pass plays with EPA > 0 allowed by the opponent defense | `0.395` |
| `opp_air_yards_allowed` | numeric | Rolling 5-game lagged mean air yards per pass attempt allowed by the opponent defense | `8.34` |
| `opp_yac_allowed` | numeric | Rolling 5-game lagged mean yards after catch per pass attempt allowed by the opponent defense | `4.88` |
 
### Quantification of Uncertainty for Numerical Features
 
The table below summarizes the observed distribution of the main numerical features in the final dataset, computed from the full pipeline output. Features marked as excluded were present in the data but withheld from the prediction model.

| Feature | Role | Mean | Std Dev | Min | 25th Pct | Median | 75th Pct | Max | Missing % |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `passing_yards` | Target | 210.377 | 94.385 | -4.000 | 157.000 | 217.000 | 274.000 | 525.000 | 0.0% |
| `days_rest` | Predictor | 29.274 | 83.111 | 4.000 | 7.000 | 7.000 | 10.000 | 1071.000 | 0.0% |
| `spread_line` | Predictor | 1.862 | 6.385 | -17.000 | -3.000 | 3.000 | 6.500 | 20.000 | 0.0% |
| `total_line` | Predictor | 44.715 | 4.556 | 28.500 | 42.000 | 44.500 | 47.500 | 58.000 | 0.0% |
| `qb_pass_share` | Predictor | 0.885 | 0.266 | 0.019 | 0.971 | 1.000 | 1.000 | 1.000 | 0.0% |
| `opp_epa_per_pass_allowed` | Predictor | 0.025 | 0.161 | -0.477 | -0.080 | 0.024 | 0.133 | 0.626 | 0.0% |
| `opp_success_rate_allowed` | Predictor | 0.451 | 0.052 | 0.273 | 0.417 | 0.451 | 0.485 | 0.617 | 0.0% |
| `opp_air_yards_allowed` | Predictor | 7.765 | 1.057 | 4.650 | 7.034 | 7.705 | 8.455 | 11.710 | 0.0% |
| `opp_yac_allowed` | Predictor | 5.223 | 0.782 | 2.813 | 4.696 | 5.185 | 5.685 | 8.657 | 0.0% |
| `bad_weather` | Predictor | 0.145 | 0.352 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0% |
| `is_home` | Predictor | 0.509 | 0.500 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.0% |
| `yards_last5` | Excluded | 213.280 | 69.276 | 1.000 | 179.950 | 225.800 | 259.525 | 383.200 | 0.0% |
| `yards_last5_std` | Excluded | 71.385 | 31.466 | 0.707 | 48.752 | 68.446 | 90.759 | 212.132 | 0.0% |
