# DS 4320 Project 2: Predicting NFL QB Passing Production

## Executive Summary

This repository contains my DS 4320 project on predicting NFL quarterback next-game passing production. I built a custom secondary dataset from public NFL source data accessed through the `nflreadpy` Python package, transformed it into nested MongoDB documents, and stored it in MongoDB Atlas. Each document represents one quarterback-game observation and includes player identity, game context, opponent context, and rolling pregame passing features tailored to the problem of projecting next-game passing yards and passing touchdowns using only information available before kickoff. The repository contains the Python scripts used to build and load the data, a Jupyter notebook analysis pipeline and markdown export, a separate press release, metadata describing the document structure and feature definitions, and background readings to help explain the project domain.

<br>

||Details|
|---|---|
| Name | Jolie Ng |
| NetID | xhh6fb |
| DOI | [![DOI - CHANGE!](https://zenodo.org/badge/DOI/10.5281/zenodo.19363443.svg)](https://doi.org/10.5281/zenodo.19363443) |
| Press Release | [Press Release Title - CHANGE!](press_release.md) |
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

### Background Reading - FIX!

The [`background_reading`](background_reading) folder contains readings that help explain the football analytics context of this project.

| Title | Brief description | File |
|---|---|---|
| nflWAR: A Reproducible Method for Offensive Player Evaluation in Football | Introduces a reproducible framework for evaluating NFL offensive players using public play-by-play data, expected points, and win probability models to estimate Wins Above Replacement (WAR). | `background_readings/01_nflwar.pdf` |
| A Reinforcement Learning Based Approach to Play Calling in Football | Formulates play calling as a sequential decision-making problem and applies reinforcement learning / expected utility optimization to identify optimal play-calling strategies based on game state. | `background_readings/02_reinforcement_learning_approach_to_play_calling.pdf` |
| NFL Play Prediction | Uses machine learning to predict the outcome of NFL plays (e.g. yardage or play result) and explores how predictive models could be used to inform or optimize in-game play selection. | `background_readings/03_nfl_play_prediction.pdf` |
| The Quarterback Prediction Problem: Forecasting NFL QB Performance | Demonstrates the difficulty of predicting NFL quarterback performance using pre-draft college and combine data, showing limited predictive power of traditional pre-draft metrics. | `background_readings/04_quarterback_prediction_problem.pdf` |
| next-gen-scraPy: Extracting NFL Tracking Data from Images to Evaluate Quarterbacks and Pass Defenses | Develops a method to extract spatial passing data from NFL Next Gen Stats visualizations and uses it to evaluate quarterback accuracy and defensive performance using spatial completion models. | `background_readings/05_next_gen_scrapy.pdf` |

| Index | Title | Brief Description | Path |
|---|---|---|---|
| 1 | Predicting the Outcome of NFL Games Using Logistic Regression | Honors thesis focused directly on NFL game outcome prediction and logistic-regression model framing | [Link](background_reading/01_predicting_outcome_of_nfl_games_using_logistic_regression.pdf) |
| 2 | Modeling NFL Football Outcomes | Paper discussing statistical models for NFL outcome prediction | [Link](background_reading/02_modeling_nfl_football_outcomes.pdf) |
| 3 | The Effect of Attendance on Home Field Advantage in the NFL | Study of home-field effects and how attendance relates to them | [Link](background_reading/03_nfl_home_field_advantage.pdf) |
| 4 | Is the NFL Betting Market Efficient? | Economics paper on whether NFL betting prices are efficient | [Link](background_reading/04_nfl_betting_market_efficiency.pdf) |
| 5 | Assessing the Convergence of the Elo Ranking Model | Paper on Elo-model convergence and ranking stability | [Link](background_reading/05_assessing_the_convergence_of_the_elo_ranking_model.pdf) |

<br>

## Data Creation
