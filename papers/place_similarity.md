# Can the Ambiance of a Place be Determined by the User Profiles of the People Who Visit It?
# Where Would You Go this Weekend? Time-Dependent Prediction of User Activity Using Social Network Data (Foursquare)
# The semantics of similarity in geographic information retrieval
very much about semantics and ontology but also describe similarity measure

Looking at http://spaceandtime.wsiabato.info/tGIS.html, result for similarity are mostly based on NLP and ontology

# Location-based and Preference-Aware Recommendation Using Sparse Geo-Social Networking Data
Collect data in two cities, find local expert in each categories, return top N location suggest by expert that are similar to the user (based on the weighted category hierarchy tree). Uses tips instead of check in. Experiment are made by keeping a test set apart.

# Location Recommendation in Location-based Social Networks using User Check-in Data
In this paper, we propose algorithms that create recommendations based on four factors: a) past user behavior (visited places), b) the location of each venue, c) the social relationships among the users, and d ) the similarity between users.
Their best method uses Personalized Page Rank on a graph whose nodes are user and location, with edges between users representing friendship and between user and location, visit

# LARS: A location-aware recommender system
LARS produces recommendations using
-  spatial ratings for non-spatial items by exploiting preference locality by user, maintaining a dynamic hierarchy of region.
-  non-spatial ratings for spatial items by using travel penalty
-  both spatial kind use both methods
