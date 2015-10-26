Code supporting my Master Thesis about finding similar neighborhood across cities using social media activity.

You can get more information by reading [our blogpost](http://northernbytes.co/2014/11/25/similar-neighborhoods/), [a two pages academic summary](http://geraud.so/neighborhoods.pdf), [our ICWSM paper](http://www.aaai.org/ocs/index.php/ICWSM/ICWSM15/paper/view/10514) or if you have more time on your hands, my [complete thesis](https://aaltodoc.aalto.fi/handle/123456789/13900).

If you're interested in our dataset, you can find it on [Figshare](http://figshare.com/articles/Foursquare_amp_Flickr_activities_in_20_cities/1584973). As Foursquare prohibits distributing out of data venue information, you will have to use `FillDB.py` to collect the latest statistics about them.

Below I provide more technical details. Yet it should be noted that for now, not all data are included with the code (although [the most important can be found on dropbox](https://dl.dropboxusercontent.com/u/23609132/cities_features_matrix.zip) and thus there is no simple demonstration one can quickly test. Hopefully this will soon be remedied :)

How it works
====================

All the code is written in Python 2 (but is known to work with automatic [2to3](https://docs.python.org/3/library/2to3.html) modifications under Python 3.4 as well) and dependencies can be install from [requirements.txt](requirements.txt). Map rendering is controlled by a Flask app and involve [Leaflet](http://leafletjs.com/) as well as some javascript in the `static` directory.

First we collect data from two social media, Foursquare and Flickr. Then we aggregate data at the venue level. Each venue become a feature vector and is stored in a matrix.

Finally we devise a method that given a polygon in one city, compute the $k$ most similar in another city.

Data collection
========================

Flickr
------------------

- `grab_photos.py` retrieve a list of all Flickr photos taken in a given city and insert them with additional metadata in a mongo database

Twitter
------------------

- `twitter.py` Listen to public twitter stream for Foursquare checkin.
- `boost_twitter.py` Gather more tweets by fetching timeline of previously discovered users

Foursquare
------------------

- `AskFourquare.py` Parse JSON Foursquare responses to relevant python object
- `CheckinAPICrawler.py` Get checkin info by requesting Foursquare API. Unfortunately, it's not working since Foursquare deployed its Swarm application, even though it can be modified to handle this new case
- `FillDB.py` Use collected tweet and `AskFourquare` to request information about users and venues, before inserting them in a Mongo database
- `FSCategories.py` Maintain a tree of Foursquare categories and provide query methods 

Data processing
========================

- `VenueFeature.py` This main step is to transform the raw data collected into a feature matrix whose rows are each venue of a city with enough visits and column are the features described in table 3 page 27.
- `Surrounding.py` Maintain a KD tree of venues to allow spatial ball query
- `FlickrVsFoursquare.py` Compute discrepancy between tweets and photos. It's not a feature associated with a single venue so it was not used in the thesis but it's interesting nonetheless. Basically, it divides a city into a grid and find the cell where the proportion of check-ins and photos are unusual. This discriminates between very touristic location that are mostly photographed (Eiffel Tower) against location where things happen (Stadium, Railway station, …)

Computation
========================

- `worldwide.py` Defines `query_in_one_city`, which performs a single similarity query between a GeoJSON polygon from one city to another
- `one_approx_query.py` Illustrate the use of `worldwide.py` on a predefined set of queries
- `approx_emd.py` Avoid computing the Earth Mover's Distance between all possible rectangles in the target city by using the pruning strategy described in section 6.1 page 38
- `ClosestNeighbor.py` Perform $k$-nearest neighbor queries over venues in two cities
- `neighborhood.py` This one does too many things and most of them turned to be not working anyway

Rendering
========================
- `ServeNN.py` A flask webserver with the following interesting routes
	* `/n/<origin>/<dest>` offer a selection of neighborhood
	* `/<origin>/<dest>/<int:knn>` interactively pick a venue in the origin city and show its $k$ nearest neighbors in the dest city 

Helper
========================

- `cities.py` Define the 20 cities we choose by their bounding box and provide methods to convert between latitude, longitude and local euclidean coordinates.
- `arguments.py`
- `calc_tsne.py`
- `Chunker.py`
- `clean_timeline.py`
- `CommonMongo.py`
- `Counter.py`
- `LocalCartesian.py`
- `OrderedDict.py`
- `persistent.py`
- `RequestsMonitor.py`
- `explore.py`
- `twitter_helper.py`
- `utils.py`

Not (directly) useful
========================
- `0708tue.py`
- `0813wed_map.py`
- `alt_emd.py`
- `bench.py`
- `CheckinCrawler.py`
- `cluster_city.py`
- `common_tag.py`
- `compare_tags.py`
- `compile_cython.py`
- `CorrectCheckIn.py`
- `extract_dataset.py`
- `extract_gold.py`
- `figure4.py`
- `first_query.py`
- `gen_status.py`
- `geom_stat.py`
- `get_brands.py`
- `ir_evaluation.py`
- `LDA.py`
- `learn_weights.py`
- `local_vs_tourist.py`
- `merge_gold.py`
- `more_query.py`
- `nldm.py`
- `outplot.py`
- `places_and_venues.py`
- `plot_corr.py`
- `plot_tag.py`
- `preprocess.py`
- `ProgressBar.py`
- `rank_disc.py`
- `read_foursquare.py`
- `report_metrics_results.py`
- `saved3.py`
- `seetags.py`
- `selection.py`
- `significance_test.py`
- `spatial_scan.py`
- `specific_emd_dst.py`
- `tag_support.py`
- `time_all_cities.py`
- `top_metrics_circle.py`
- `VenueIdCrawler.py`
- `wordplot.py`
- `emd_leftover.py`
