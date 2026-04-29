use project2_db

// sample documents
db.qb_games.find().limit(5)

// filter QB
db.qb_games.find({
  "player_info.player_name": "Patrick Mahomes"
})

// high yard games
db.qb_games.find({
  "targets.passing_yards": { $gt: 300 }
})

// rolling performance check
db.qb_games.find({
  "pregame_form.yards_last3": { $gt: 250 }
})

// aggregation: average yards per QB
db.qb_games.aggregate([
  {
    $group: {
      _id: "$player_info.player_name",
      avg_yards: { $avg: "$targets.passing_yards" }
    }
  },
  { $sort: { avg_yards: -1 } }
])
