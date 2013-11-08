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
    USA: 462718 in 64636.151047s
    2013-10-23 11:39:47,532 [INFO]: Saved a total of 632673 photos.

Usefull site:

-  [http://geojsonlint.com/](http://geojsonlint.com/)

-  [http://jsonlint.com/](http://jsonlint.com/)

-  to get cartographic shapefiles [http://mapserver.flightgear.org/shpdl/](http://mapserver.flightgear.org/shpdl/)

In San Fransisco, there are 5916822 tags for 787660 photos (145608 unique),
44354 appearing at least 5 times.

-----------   --------    ------   --------  --------  --------  --------  --------
rank            1            5        10         50      100        1000    44354
proportion     7.3645     15.582    21.734    32.207    37.655    63.044    97.345
-----------   --------    ------   --------  --------  --------  --------  --------

![Tags distribution](tags_sf.pdf)


--------------------
sanfrancisco
california
iphoneography
square
squareformat
instagramapp
unitedstates
sf
usa
ca
san
francisco
goldengatepark
2010
iphone
--------------------

: First 15 tags

----------------------------------------
2013
pacific
february
foundinsf
dolorespark
japaneseteagarden
boat
5k
national
σανφρανσίσκο
cruise
above
july2009
effortlesslyuploadedbymyeyeficard
dayofdecision
----------------------------------------

: 15 random tags between 100 and 1000

----------------------------------------
sfgiantsfan
rolexbigboatseries
proshowgold
neutraface
natur€
lusty
lightousetender
jennyholzer
img0562jpg
djguyruben
cutebaby
cardamine
californiaproduce
aroundwithb1
aquateenhungerforcemooninite
----------------------------------------

: 15 random tags between 90000 and 140000


pairwise distance for museum vs street

entropy.dat (some filtering, suspicous value)
sfmoma
14thstreet: 
[('22647597@N03', 3550),
 ('57453294@N00', 18),
 ('28198184@N04', 18),
 ('17607628@N00', 16),
 ('29309361@N00', 11),
 ('27219489@N07', 3),
 ('79781814@N00', 2),
 ('88077630@N00', 1),
 ('68202512@N00', 1),
 ('45750259@N05', 1),
 ('13563767@N03', 1),
 ('13319346@N05', 1)]

heatmap.png (log of count)
season.png (hard to see ≠)

tourist:
tourists proportion: 65%
tourists' photos proportion: 19%
tourist 90 percentile: 21.0
local 90 percentile: 148.0
filter "anomalous" locals?

map:
tourist/total - 0.5 ∈
[-0.5 (only local, blue), … 0 (neutral, white), … 0.5 (only tourist, red)]

top tag intersection

# Friday 8
function duplicateTags(){
    db.photos.find({hint: "sf"}).forEach(function(doc){
         db.photos.update({_id:doc._id}, {$set:{"ntags":doc.tags}});
    });}

Pre processing used to remove spurious photos:
We want to get rid of burst of photos (say more than a threshold T) taken by
the same user, in the same place around the same time.
```
First make a loop to duplicates "tags" of every photo.
for each user u
	get a list L of the tags she has used more than T times
		for each tag t of L
			get its distribution in quantized space (200x200) and time (2 weeks)
			get index of the corresponding count matrix > T
			update those photos id by setting ntags = tags - t
			keep a count of how many photos were cleaned
```
recompute tag usage: out.patch
recompute nentropies.dat and KL w.r.t background distribution nKentropies.dat
compute gravity distance
zathura tag_cloud.pdf &
and entropy e_grav.dat
compute pairwise distance entropy e_pair.dat
compute time entropy time_entropy.txt
LANG=C soffice time_entropy.ods &
dimensionality reduction
zathura red.pdf &
other course

What we discussed:

- preprocessing the tag. For each user u and each tag t, I computed the
  distribution of photos tagged t by u and removed the tags from those photos
that appear more than T times in the same place (in the 200×200 discrete grid)
in the same time (2 weeks interval). For T=120, it removed around 20% of all
tags and it somewhat changed the list of top tag (see out.patch) but with no
clear pattern. At least, it removed very low entropy value.

- KL divergence. I computed D(tag distribution||photos distribution) >= 0
  which is indeed low for "general" tag like sanfrancisco or california and
higher for tag like ucsfschoolofdentistry (see nKentropies.dat)

- Gravity. For the top tags, I computed the distribution of distance between
  each photo and the gravity center of the tag. Then I plotted these tags with
the mean distance in the horizontal axis and standard deviation in vertical
axis (see tag_cloud.pdf, where the point are bluer when regular entropy is
larger). It seems that there is relationship between mean and standard
deviation which in my opinion means that as photos are farther from their
center, they tend to be uniformly distributed in that larger area. (or at
least with no clear motif) But it's not clear whether we can take advantage of
that (rather natural) fact.

- Time entropy. I computed entropy in time for the top tag at different scale
  (day, week, month, quarter, year) and again, some tags with low day entropy
are event (baytobreaker, googleio, wwdc, karnaval, sfpride) while general tags
have high entropy (see
https://docs.google.com/spreadsheet/ccc?key=0AtQDypHHoV_OdGhwcVVaZGZZLWRRWF8yR3otRzY5dnc&usp=sharing).

Then we agree that we have done enough exploration and that in the last 4 weeks, we should focus on two problems:

- given a tag, find the top k location where it is concentrated, using for
  instance spatial scan:
http://www.cs.utah.edu/~jeffp/papers/stat-disc-KDD06.pdf

- conversely, given a location, find top k tags that best describe it compared
  with other locations of similar scale.
