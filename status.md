city         | flickr   | venues  | users
newyork      |          |         | 
washington   |          |         | 
sanfrancisco |          |         | 
atlanta      |          |         | 
indianapolis |          |         | 
losangeles   |          |         | 
seattle      |          |         | 
houston      |          |         | 
stlouis      |          |         | 
chicago      |          |         | 
london       |          |         | 
paris        | 159,969  | 7,955   | 
berlin       |          |         | 
rome         |          |         | 
prague       |          |         | 
moscow       |          |         | 
amsterdam    |          |         | 
helsinki     | 21,333   | 1,563   | 
stockholm    | 26,540   | 2,648   | 
barcelona    | 98,730   | 4,932   | 

`db.photos.aggregate([{$project: {hint: 1}}, {$group: {_id: '$hint', count: {$sum: 1}}}])`
`db.venue.aggregate([{$project: {city: 1}}, {$group: {_id: '$city', count: {$sum: 1}}}])`
