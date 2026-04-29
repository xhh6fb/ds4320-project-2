// switch database
use project2_db

// view documents
db.qb_games.find().limit(5)

// filter by QB
db.qb_games.find({
  "player_info.player_name": "Patrick Mahomes"
})

// high yard games
db.qb_games.find({
  "targets.passing_yards": { $gt: 300 }
})

// aggregation example
db.qb_games.aggregate([
  {
    $group: {
      _id: "$player_info.player_name",
      avg_yards: { $avg: "$targets.passing_yards" }
    }
  },
  { $sort: { avg_yards: -1 } }
])
