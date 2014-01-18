# Towards Automated Georeferencing of Flickr Photos

Problem: localize photo in and within city given tags

First perform k-medoids (better than k-means?) to identify region

Filter place tags by high chi2 score

Keep only the best and train a multinomial naive bayes classifier

Good enough to find the correct a 1 or 2 km radius around the real location

Can be marginally improved by smoothing

Probability distribution over the powerset of an universe: Dempster-Shafer theory
