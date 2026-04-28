use ds4320_project2

db.qb_game_docs.countDocuments()

db.qb_game_docs.findOne()

db.qb_game_docs.find(
  { season: 2024, "player_info.team": "BUF" },
  { "player_info.player_name": 1, week: 1, targets: 1 }
).limit(5)

db.qb_game_docs.aggregate([
  {
    $group: {
      _id: "$season",
      avg_pass_yards: { $avg: "$targets.passing_yards" },
      avg_pass_tds: { $avg: "$targets.passing_tds" },
      n_docs: { $sum: 1 }
    }
  },
  { $sort: { _id: 1 } }
])
