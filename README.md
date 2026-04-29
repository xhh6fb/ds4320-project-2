# DS 4320 Project 2: Predicting NFL QB Passing Production

## Executive Summary
 
This repository contains my DS 4320 project on predicting NFL quarterback next-game passing yards using only structural, contextual, and opponent-facing pregame features — deliberately excluding a quarterback's own historical passing yardage as a model input. I built a custom secondary dataset from public NFL source data accessed through the `nfl_data_py` Python package, transformed it into nested MongoDB documents, and stored it in MongoDB Atlas. Each document represents one quarterback-game observation and includes player identity, game context (including betting market lines), opponent defensive quality metrics derived from EPA and advanced pass-defense statistics, weather, usage share, and rolling pregame features, all constructed to contain zero information from the current game. The research question is whether structural pregame information (game environment, opponent defense, and usage role) can meaningfully predict passing volume without feeding the model how many yards that quarterback threw for last week. The modeling pipeline compares a mean baseline, Ridge regression, Random Forest, and XGBoost against a 2024 holdout season; XGBoost achieved the best performance at MAE 62.51 yards and R² of 0.22, representing an 11% reduction in error over the baseline. The repository contains the Python scripts used to build and load the data, a Jupyter notebook analysis pipeline and markdown export, a separate press release, metadata describing the document structure and feature definitions, and background readings explaining the project domain.

<br>

||Details|
|---|---|
| Name | Jolie Ng |
| NetID | xhh6fb |
| DOI | [![DOI - CHANGE!](https://zenodo.org/badge/DOI/10.5281/zenodo.19363443.svg)](https://doi.org/10.5281/zenodo.19363443) |
| Press Release | [Can Context Alone Predict a Quarterback's Passing Day?](press_release.md) |
| Pipeline | [Notebook](pipeline/project2_pipeline.ipynb) & [Markdown](pipeline/project2_pipeline.md)   |
| License | [MIT](LICENSE) |

<br>

## Problem Definition
 
### Initial General Problem & Refined Specific Problem Statement
 
**Initial General Problem:** Projecting athletic performance.
 
**Refined Specific Problem Statement:** Build a document-model NFL dataset in MongoDB using nflverse data and use it to project next-game passing yards for NFL quarterbacks, specifically testing whether structural pregame features (betting market lines, opponent pass-defense quality, weather, usage share, and scheduling context) can produce accurate predictions *without* relying on the quarterback's own historical passing yard totals as input features.

### Motivation
 
Quarterback performance is arguably the single most consequential variable in determining NFL game outcomes, and accurately projecting it in advance has enormous value across domains including team analytics, sports media, and fantasy sports. Most prediction approaches lean heavily on a quarterback's recent statistical history, treating last week's yards as the strongest signal for next week's. But this creates a practical blind spot: a newly installed starter, a quarterback returning from injury, or a player who just switched teams may have little or no usable history. More fundamentally, raw yard totals conflate many underlying factors (opponent difficulty, game script, team pass volume, and weather) that could be modeled directly from information available before kickoff. This project tests whether those structural factors, captured through tools like EPA-based opponent defense ratings, betting market total lines, and each quarterback's share of their team's pass volume, contain enough signal on their own to project passing production meaningfully. The answer has practical implications: a model that works without historical yards is useful precisely where history-based models fail.
 
### Rationale for Refinement
 
I refined the problem from broad athletic performance prediction to a targeted test of whether contextual features alone can predict quarterback passing yards. The general problem is too wide: it spans multiple sports, positions, and target variables, and the most obvious modeling approach (feed in recent stats, get a prediction) answers a different, much easier question. The interesting and harder question is what happens when you take the historical yard totals away. If you can still predict passing production reasonably well from environment alone, that tells you something important: game context and opponent quality are doing most of the explanatory work, not the quarterback's individual talent signal contained in recent yardage. That is a more analytically useful finding, and it produces a model that generalizes better to situations without a useful history and generates more interpretable insights about what structural drivers matter.
 
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
| EPA | Expected Points Added | A play-level measure of how much a play improves or harms a team's expected scoring. Used here to measure opponent pass-defense quality before the game. |
| Success rate | Share of pass plays where EPA > 0 | An opponent-defense quality metric measuring how often the defense shuts down passing plays, regardless of yards gained. |
| Air yards | Yards the ball travels through the air from the line of scrimmage to the catch point | A pregame opponent metric indicating how far downfield the defense allows the passing game to operate. |
| Yards after catch (YAC) | Yards a receiver gains after catching the ball | A pregame opponent metric indicating the defense's ability to limit yards after contact. |
| QB pass share | Player pass attempts divided by total team pass attempts in the same game | A usage metric capturing how much of the team's passing volume runs through the starting quarterback. |
| Spread line | The point spread set by oddsmakers for the game | A market-based pregame signal for expected game script; large spreads imply the trailing team will throw more to catch up. |
| Total line | The over/under points total set by oddsmakers | A market-based signal for expected offensive volume in the game overall. |
| Pregame feature | Any variable known before kickoff | All input features in this model must be pregame features to ensure there is no data leakage from the current game. |
| Rolling average | An average calculated from prior games only, with a one-game lag applied | Used to summarize opponent pass-defense trends without including current-game information. |
| Leakage | Using current-game information as a model input | A methodological error that makes the model unrealistically accurate in training but useless in real prediction. |
| Game script | The in-game strategic tendency driven by score and game state | Trailing teams throw more; market lines capture pre-game expectations of game script. |
| Home/away split | Whether a game is played at the QB's home stadium or away | Crowd noise, travel fatigue, and scheduling factors can influence performance. |
| Rest days | Days since the player's previous game | Short rest (Thursday games) can affect performance; included as a scheduling feature. |
| Bad weather | Binary flag for wind > 15 mph or temperature < 32°F | Extreme weather suppresses passing volume and makes the passing game harder to execute. |
| Document model | A database structure that stores records as nested JSON-like documents | Used to store each quarterback-game observation as a self-contained document with nested context subfields. |
| MongoDB | A document-oriented database system | The storage system used for the custom quarterback-game dataset. |
| nflverse | An open NFL analytics data ecosystem | The source of all raw data used to build the custom secondary dataset. |
| nfl_data_py | The Python package used to access nflverse play-by-play data | The tool used to load raw NFL play-by-play tables into Python for cleaning, aggregation, and feature engineering. |
| XGBoost | Extreme Gradient Boosting — a tree-based machine learning algorithm | The best-performing model in the pipeline, achieving MAE of 62.51 yards on the 2024 holdout season. |
| MAE | Mean Absolute Error — average absolute difference between predicted and actual values | The primary evaluation metric for model comparison; reported in yards. |
| R² | Coefficient of determination — proportion of variance in the target explained by the model | A model fit measure where 1.0 is perfect and 0.0 matches the mean-only baseline. |
 
### Domain
 
This project lives in the domain of sports analytics, with a specific focus on NFL offensive performance modeling. Sports analytics uses historical game and player data to answer questions about evaluation, prediction, and strategy. Within that domain, quarterback passing projection is one of the most studied problems because passing production directly drives scoring and game outcomes. This project takes a non-standard approach by restricting the model's inputs to structural, contextual features (opponent pass-defense quality, betting market lines, usage share, weather, and scheduling) and excluding the quarterback's own historical passing yardage. The domain-specific insight motivating this design is that game context explains a large share of passing volume variance: a quarterback facing a weak defense in a high-total game with expected negative game script will throw more, regardless of what they did last week. The modeling results confirm this hypothesis produces real predictive signal (XGBoost achieved an 11% MAE improvement over the baseline) while also revealing the limits of the approach and where further structure could help.
 
### Background Reading

My OneDrive folder [DS4320 Project 2 Background Reading](https://myuva-my.sharepoint.com/:f:/g/personal/xhh6fb_virginia_edu/IgC-Mn_orQyjRp3d7qAwHAhNAYPykaYrIQBMBeFxQXjiRuw?e=ZiLZca) contains readings explaining the football analytics context of this project.
For convenience, I also have the readings stored in this Github repo in the [`background_reading`](background_reading) folder.

| Index | Title | Brief Description | Github Path | OneDrive Path |
|---|---|---|---|---|
| 1 | nflWAR: A Reproducible Method for Offensive Player Evaluation in Football | Reproducible NFL offensive player evaluation using WAR models | [Link](background_reading/01_nflwar.pdf) | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/xhh6fb_virginia_edu/IQBveOzDivz-ToCnxDH7yWIkAQSwZw88qENRVEIs1yBqv-Q?e=dMNfYf) |
| 2 | A Reinforcement Learning Based Approach to Play Calling in Football | Uses reinforcement learning to optimize football play-calling decisions | [Link](background_reading/02_reinforcement_learning_approach_to_play_calling.pdf) | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/xhh6fb_virginia_edu/IQBO0102i6QkR6WlO614cE8GAYW_TAGzv-aIXWM85DI2IQY?e=sZEY8s) |
| 3 | NFL Play Prediction | Machine learning models predicting NFL play outcomes and yardage | [Link](background_reading/03_nfl_play_prediction.pdf) | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/xhh6fb_virginia_edu/IQDkVx9F1ETNTKlmygqcTWKxAXSdOl_07PzybnKjfPLqqkg?e=RM2biF) |
| 4 | The Quarterback Prediction Problem: Forecasting NFL QB Performance | Illustrates the difficulty of predicting QB performance from limited or pre-draft data | [Link](background_reading/04_quarterback_prediction_problem.pdf) | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/xhh6fb_virginia_edu/IQBcoltTSCQZR6fGZx43-3dQAYFkuYyUX3S_Z0ETRLv9Leo?e=fBwaZJ) |
| 5 | next-gen-scraPy: Extracting NFL Tracking Data to Evaluate Quarterbacks and Pass Defenses | Extracts tracking data to evaluate quarterbacks and pass defenses using air yards and route metrics | [Link](background_reading/05_next_gen_scrapy.pdf) | [Link](https://myuva-my.sharepoint.com/:b:/g/personal/xhh6fb_virginia_edu/IQADVfgKtjPPQa0oFO7cEBmEAVxUZOK-9DkI1azoQnXRBpY?e=oJlO96) |

<br>

## Data Creation
 
### Provenance
 
The raw data for this project comes from the nflverse ecosystem, a public collection of NFL analytics data and tooling. In Python, I access the data through the `nfl_data_py` package, which is the maintained Python interface for loading nflverse datasets. The primary raw input is play-by-play data covering the 2020 through 2024 NFL regular seasons, loaded via `nfl.import_pbp_data()`. Each row in the play-by-play table represents one play and includes the passer, teams, outcome, EPA, air yards, yards after catch, weather conditions, betting lines, and dozens of other contextual fields.
 
From this raw table, I construct two types of aggregations. The first is a quarterback-game aggregation that produces one row per quarterback per game, summing passing yards and attempts and taking the first observed value for game-level constants like weather and market lines. The second is a defensive aggregation that computes, for each team-game on defense, the average EPA per pass attempt, success rate, air yards allowed, and YAC allowed — then rolls those metrics over the prior five games with a one-game lag so that no current-game information enters the features. The two aggregations are joined on season, week, and opponent to produce the final analysis-ready dataset. A separate team-level pass attempt aggregation is computed to derive the `qb_pass_share` usage feature. The final dataset is a secondary, purpose-built dataset created from public raw source data rather than directly downloaded or manually collected data.
 
### Data Creation Code
 
| File | Description | Path |
|---|---|---|
| `src/build_project2.py` | Loads raw nflverse play-by-play data, aggregates to QB-game level, computes team-level pass attempts for usage share, merges betting line and weather features, calls the utils feature pipeline, joins defensive context, and writes the final CSV | [`src/build_project2.py`](src/build_project2.py) |
| `src/utils_project2.py` | Helper functions for logging, QB rolling form, rest days, weather flags, home/away indicator, and the rolling opponent pass-defense feature pipeline | [`src/utils_project2.py`](src/utils_project2.py) |
| `pipeline/project2_pipeline.ipynb` | Queries MongoDB into a dataframe, runs the modeling pipeline excluding historical yards features, evaluates predictions, and generates visualizations | [`pipeline/project2_pipeline.ipynb`](pipeline/project2_pipeline.ipynb) |
| `pipeline/project2_pipeline.md` | Markdown export of the notebook pipeline | [`pipeline/project2_pipeline.md`](pipeline/project2_pipeline.md) |
 
### Bias Identification
 
Bias can enter the data collection and feature engineering process in several ways. First, the dataset only includes game observations where a quarterback threw enough passes to be identified as the primary passer, meaning backup quarterbacks with very low attempt volumes may be excluded by the `dropna()` filter applied after the rolling window calculations. Second, quarterbacks on pass-heavy offenses have systematically higher `qb_pass_share` values, so the usage feature implicitly captures team-level play-calling tendencies in addition to individual workload. Third, the rolling opponent defense metrics reflect the schedule of opponents each defense has faced, so a defense that plays weak offenses early in the season will appear stronger than it truly is. Fourth, the `days_rest` feature has a maximum of 1,071 days in the dataset, corresponding to quarterbacks returning from multi-season absences such as retirement or extended injury; these observations carry no meaningful scheduling signal and may distort the feature distribution. Fifth, by excluding historical passing yards as a predictor, the model may systematically underperform for quarterbacks whose volume is highly consistent or highly volatile, since those tendencies cannot be captured without the autoregressive signal.
 
### Bias Mitigation
 
I mitigate these biases through several design choices. The rolling opponent features use a five-game lag window with a one-game shift, which smooths out any single-game anomaly and prevents current-game leakage. The `min_periods=1` rolling parameter prevents early-season rows from being dropped purely due to insufficient history. Including betting market lines (spread and total) as features partially corrects for opponent schedule bias, since market makers implicitly account for strength of schedule when setting lines. The temporal train/test split (2020–2023 train, 2024 test) ensures the model is evaluated on a future holdout season rather than a random sample, which better reflects real-world forecasting conditions. For the `days_rest` outliers, future pipeline iterations should cap rest at a maximum of 365 days, as any gap beyond one full season provides no useful scheduling information. In reporting model results, performance is compared across all three non-baseline models to confirm that findings are not specific to a single algorithm.
 
### Rationale for Critical Decisions
 
The most consequential decision in this project is the exclusion of historical passing yards as a direct predictor. This was a deliberate research design choice rather than a data availability constraint. Most naive models for projecting passing yards use `yards_last_n` as their strongest feature, but this conflates predicting performance with extrapolating an autocorrelated time series. Excluding it forces the model to use features that explain *why* a quarterback might have a high-volume game (game script expectations from the spread, offense-defense matchup quality from EPA metrics, and usage role from pass share) rather than simply regressing on what happened before. This produces a model that generalizes better to situations without a useful history and generates more interpretable insights about what structural drivers matter.
 
A second major decision is using play-by-play data rather than pre-aggregated box score data. This allows the defensive features to be computed correctly at the play level (filtering to pass attempts only before aggregating EPA, air yards, and YAC) which is more accurate than applying those metrics to all play types. A third decision is to sort defensive rolling features by `(defteam, season, week)` rather than just `(defteam, week)`, which is necessary to ensure correct temporal ordering across season boundaries. A fourth decision is to use a temporal holdout split (2024 as test) rather than random cross-validation, which is essential for time-series data where random sampling would allow future games to appear in training.
 
<br>

## Metadata
 
### Implicit Schema
 
The MongoDB collection stores one document per quarterback-game observation. Each document represents the quarterback's full pregame situation for a specific game, along with the actual game outcome used for prediction evaluation. The structure follows a consistent nested schema across all documents. In my [pipeline notebook](pipeline/project2_pipeline.ipynb), 1632 documents were inserted into MongoDB.
 
#### Top-level document fields
 
- `_id`: custom unique identifier combining game_id and player_id
- `game_id`: source game identifier (e.g., `2024_03_NE_NYJ`)
- `season`: NFL season year
- `week`: NFL week number
- `game_date`: calendar date of the game
#### Nested object: `player_info`
 
- `player_id`: player identifier string
- `player_name`: quarterback display name
- `team`: quarterback's team abbreviation for that game
#### Nested object: `game_context`
 
- `opponent`: opposing team abbreviation
- `home_team`: home team abbreviation
- `away_team`: away team abbreviation
- `is_home`: binary indicator (1 = home, 0 = away)
- `days_rest`: days since the quarterback's previous game
- `temp`: temperature in degrees Fahrenheit at game time
- `wind`: wind speed in mph at game time
- `bad_weather`: binary flag (1 = wind > 15 mph or temp < 32°F)
- `time_of_day`: kickoff timestamp string
- `spread_line`: pregame point spread (negative favors home team)
- `total_line`: pregame over/under points total
#### Nested object: `usage_context`
 
- `pass_attempts`: quarterback's total pass attempts in the game
- `team_attempts`: total team pass attempts in the game
- `qb_pass_share`: quarterback's pass attempts divided by team attempts
#### Nested object: `pregame_form`
 
Present in the dataset but intentionally excluded as model predictors.
 
- `yards_last5`: rolling 5-game average passing yards (lagged — does not include current game)
- `yards_last5_std`: rolling 5-game standard deviation of passing yards (lagged)
#### Nested object: `opponent_context`
 
All metrics lagged and rolled over the prior five games to prevent leakage.
 
- `opp_epa_per_pass_allowed`: rolling 5-game mean EPA per pass allowed
- `opp_success_rate_allowed`: rolling 5-game mean pass-play success rate allowed
- `opp_air_yards_allowed`: rolling 5-game mean air yards per pass attempt allowed
- `opp_yac_allowed`: rolling 5-game mean yards after catch per attempt allowed
#### Nested object: `targets`
 
- `passing_yards`: total passing yards in the game — **prediction target**
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
| Data Source | nflverse via `nfl_data_py` Python package |
| Seasons Covered | 2020–2024 NFL seasons |
| Unit of Observation | One quarterback-game |
| Total Observations | 1,762 rows (pre-target creation); 1,632 after dropping last game per QB |
| Training Set | 2020–2023 seasons |
| Test Set | 2024 season (temporal holdout) |
| Database | MongoDB Atlas |
| Database Name | `project2_db` |
| Collection | `qb_games` |
| Main Dataset File | [`data/qb_games.csv`](data/qb_games.csv) |
| Primary Target Variable | `passing_yards` (next game) |
| Model Predictors | Opponent defense (EPA, success rate, air yards, YAC), market lines (spread, total), usage (qb_pass_share), scheduling (days_rest, is_home), weather (temp, wind, bad_weather) |
| Excluded from Prediction | QB's own historical passing yards (`yards_last5`, `yards_last5_std`) |
| Best Model | XGBoost — MAE 62.51 yds, RMSE 79.25 yds, R² 0.2226 |
| Baseline MAE | 70.36 yards (train mean prediction) |
 
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
| `qb_pass_share` | numeric | QB's pass attempts divided by team attempts | `1.0` |
| `yards_last5` | numeric | Rolling 5-game average passing yards from prior games only — **excluded as model predictor** | `141.4` |
| `yards_last5_std` | numeric | Rolling 5-game standard deviation of passing yards from prior games only — **excluded as model predictor** | `80.9` |
| `prev_game_date` | date | Date of the quarterback's previous game | `2024-09-15` |
| `days_rest` | numeric | Days between the previous game and the current game | `4` |
| `bad_weather` | integer | Binary flag: 1 if wind > 15 mph or temp < 32°F, else 0 | `0` |
| `is_home` | integer | Binary flag: 1 if quarterback's team is the home team, else 0 | `1` |
| `opp_epa_per_pass_allowed` | numeric | Rolling 5-game lagged mean EPA per pass attempt allowed by the opponent defense | `-0.180` |
| `opp_success_rate_allowed` | numeric | Rolling 5-game lagged mean rate of pass plays with EPA > 0 allowed by the opponent defense | `0.395` |
| `opp_air_yards_allowed` | numeric | Rolling 5-game lagged mean air yards per pass attempt allowed by the opponent defense | `8.34` |
| `opp_yac_allowed` | numeric | Rolling 5-game lagged mean yards after catch per attempt allowed by the opponent defense | `4.88` |
 
### Quantification of Uncertainty for Numerical Features
 
The table below reports the observed distribution of all numerical features in the final dataset, computed from the full pipeline output on 1,762 quarterback-game observations. Features marked Excluded were present in the data but withheld from all models. Two distributional anomalies are worth noting: `days_rest` has a maximum of 1,071 days (quarterbacks returning from multi-season absences, which is a real value but one carrying no scheduling signal, and a candidate for capping in future iterations), and `passing_yards` has a minimum of -4 (a valid NFL outcome where a receiver loses yards after catching a short pass, not a data error).

| Feature | Role | Mean | Std Dev | Min | 25th Pct | Median | 75th Pct | Max | Missing % |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `passing_yards` | Target | 210.38 | 94.39 | -4.000 | 157.000 | 217.000 | 274.000 | 525.000 | 0.0% |
| `days_rest` | Predictor | 29.27 | 83.11 | 4.000 | 7.000 | 7.000 | 10.000 | 1071.000 | 0.0% |
| `spread_line` | Predictor | 1.86 | 6.39 | -17.000 | -3.000 | 3.000 | 6.500 | 20.000 | 0.0% |
| `total_line` | Predictor | 44.72 | 4.56 | 28.500 | 42.000 | 44.500 | 47.500 | 58.000 | 0.0% |
| `qb_pass_share` | Predictor | 0.885 | 0.266 | 0.019 | 0.971 | 1.000 | 1.000 | 1.000 | 0.0% |
| `opp_epa_per_pass_allowed` | Predictor | 0.025 | 0.161 | -0.477 | -0.080 | 0.024 | 0.133 | 0.626 | 0.0% |
| `opp_success_rate_allowed` | Predictor | 0.451 | 0.052 | 0.273 | 0.417 | 0.451 | 0.485 | 0.617 | 0.0% |
| `opp_air_yards_allowed` | Predictor | 7.765 | 1.057 | 4.650 | 7.034 | 7.705 | 8.455 | 11.710 | 0.0% |
| `opp_yac_allowed` | Predictor | 5.223 | 0.782 | 2.813 | 4.696 | 5.185 | 5.685 | 8.657 | 0.0% |
| `bad_weather` | Predictor | 0.145 | 0.352 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.0% |
| `is_home` | Predictor | 0.509 | 0.500 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.0% |
| `yards_last5` | Excluded | 213.28 | 69.28 | 1.000 | 179.950 | 225.800 | 259.525 | 383.200 | 0.0% |
| `yards_last5_std` | Excluded | 71.39 | 31.47 | 0.707 | 48.752 | 68.446 | 90.759 | 212.132 | 0.0% |
