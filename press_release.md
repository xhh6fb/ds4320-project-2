# New NFL Document Database Uses Pregame Trends to Project Quarterback Performance

## Hook

What is an NFL quarterback likely to do next week? That question matters to fans, fantasy players, analysts, and anyone interested in understanding game performance before kickoff. In this project, I built a custom document-model NFL dataset that uses pregame trends to estimate a quarterback’s next-game passing yards and passing touchdown production.

## Problem Statement

Quarterback passing production is one of the most important and visible parts of NFL performance, but it is also difficult to predict. A quarterback’s passing yards and touchdowns can change from week to week because of recent form, usage, opponent strength, game location, and scheduling context. Public football data exists, but it is usually not already organized for this exact prediction problem. In particular, many raw tables are not built to answer a clean pregame forecasting question, and they often need substantial filtering and feature engineering before they can support a model.

My specific problem in this project was to create a secondary dataset that could support prediction of **next-game passing yards** and **next-game passing touchdowns** for NFL quarterbacks using only information available before the game started.

## Solution Description

To solve this problem, I used publicly available nflverse data accessed through the `nflreadpy` Python package and transformed it into a custom MongoDB document database. Instead of treating the raw online tables as the final dataset, I created a new quarterback-game dataset specifically tailored to the question I wanted to answer.

Each document in the database represents one quarterback-game observation. The documents include nested information about the quarterback, the game context, the opponent, and the quarterback’s recent passing form. I also engineered pregame features such as rolling averages of passing yards, passing touchdowns, attempts, completions, and opponent pass-defense context. That makes the final dataset more directly useful for predictive modeling than a generic source table.

After building and loading the documents into MongoDB Atlas, I created a separate Python notebook pipeline that queried the database into a dataframe, trained machine learning models, and evaluated how well those models could predict passing yards and passing touchdowns. The goal of the project was not to claim perfect prediction, but to show that carefully prepared pregame features in a document-model dataset can support a meaningful quarterback projection workflow.

## Chart

Insert the project chart below. A good choice is the scatter plot of actual versus predicted passing yards on the test set.

![Actual vs Predicted Passing Yards](figures/qb_prediction_results.png)
