Since 2008-08-01,

-  in San Fransisco SFBL = (37.7123, -122.531) SFTR = (37.7981, -122.364): 816462

-  in New-York NYBL = (40.583, -74.040) NYTR = (40.883, -73.767): 1817663

-  in London LDBL = (51.475, -0.245) LDTR = (51.597, 0.034): 1614306

-  in Virginia VGTR = (38.62, -76.27) VGBL = (36.80, -78.52): 190817

-  in California CABL = (37.05, -122.21) CATR = (39.59, -119.72): 784313

3085 photos for the following:

```curl "http://api.flickr.com/services/rest/?min_upload_date=1199145600&format=json&min_taken_date=1990-07-18+17%3A00%3A00&nojsoncallback=1&method=flickr.photos.search&extras=date_upload%2Cdate_taken%2Cgeo%2Ctags&bbox=-122.395%2C37.771%2C-122.387%2C37.777&content_type=1&media=photos&per_page=1&page=1&accuracy=16"```

recursive call

-  at level 0 ((37.7123, -122.531) -- (37.7981, -122.364)): 790275

-  at level 1 ((37.7552, -122.4475) -- (37.7981, -122.364)): 608840

-  at level 2 ((37.7552, -122.40575) -- (37.77665, -122.364)): 35522

-  at level 3 ((37.765925, -122.40575) -- (37.77665, -122.384875)): 18107

-  at level 4 ((37.765925, -122.40575) -- (37.7712875, -122.3953125)): 4035




    2013-10-16 21:02:45,725 [INFO]: Finish ((37.768, -122.4), (37.778, -122.38)):
    11333 photos in 1369.99082398s
    flickr reports 14177 total
    2013-10-16 21:02:45,726 [INFO]: Insert 10588 photos (7% duplicate).
    2013-10-16 21:02:45,726 [INFO]: made 188 requests.

27620 per hour
662900 per day

counting tags over 10588 photos

aggregate, shell:

    db.photos.aggregate([ {"$unwind": "$tags"},
    {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
    {"$sort": {"count": -1, "_id": -1}} ])
126ms

map reduce, shell:

    var mapF = function () { this.tags.forEach(function(z) { emit(z, 1); }); }
    var redF = function (key, values) { var total = 0;
    for (var i = 0; i < values.length; i++)
    { total += values[i]; } return total; }
    db.photos.mapReduce( mapF, redF, { out: "tag_count" })
1170ms

aggregate, python:

	70.807ms ({u'count': 6405, u'_id': u'sanfrancisco'})
map reduce, python:

	879ms ({u'_id': u'sanfrancisco', u'value': 6405.0})

San Fransisco:

	2013-10-17 15:29:00,254 [INFO]: Finish ((37.7123, -122.531), (37.7981, -122.364)):
	659091 photos in 54419.3643861s
	2013-10-17 15:29:00,255 [INFO]: Saved a total of 659091 photos.
	2013-10-17 15:29:00,255 [INFO]: made 2922 requests.
	> db.photos.count()
	641142
	2.72% duplicate
	42414 per hour

California:

	2013-10-18 12:39:56,312 [INFO]: Finish ((37.05, -122.21), (39.59, -119.72)):
	612050 photos in 48558.9884s
	2013-10-18 12:39:56,313 [INFO]: Saved a total of 612050 photos.
	2013-10-18 12:39:56,313 [INFO]: made 2250 requests.
	> db.photos.count() - 641142
	594176
	2.92% duplicate
	44050 per hour

Usefull site:

-  [http://geojsonlint.com/](http://geojsonlint.com/)

-  [http://jsonlint.com/](http://jsonlint.com/)

-  to get cartographic shapefiles [http://mapserver.flightgear.org/shpdl/](http://mapserver.flightgear.org/shpdl/)
