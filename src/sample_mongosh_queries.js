use ds4320_project2

db.qb_game_docs.countDocuments()

db.qb_game_docs.findOne()

db.qb_game_docs.find(
  { season: 2024 },
  {
    "player_info.player_name": 1,
    "player_info.team": 1,
    week: 1,
    "targets.passing_yards": 1,
    "targets.passing_tds": 1
  }
).limit(5)

db.qb_game_docs.find(
  { "player_info.player_name": "Josh Allen" },
  {
    season: 1,
    week: 1,
    "pregame_form.avg_pass_yards_last_3": 1,
    "targets.passing_yards": 1
  }
).sort({ season: 1, week: 1 }).limit(10)

db.qb_game_docs.aggregate([
  {
    $group: {
      _id: "$season",
      doc_count: { $sum: 1 },
      avg_passing_yards: { $avg: "$targets.passing_yards" },
      avg_passing_tds: { $avg: "$targets.passing_tds" }
    }
  },
  { $sort: { _id: 1 } }
])
