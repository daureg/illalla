# Geographical Topic Discovery and Comparison

Presents location-driven model, text-driven model, and a new combined model LGTA to:

- discover geo coherent topics

- compare topics across locations

## Location driven

cluster photos using k-means, DBSCAN, mean-shift or mixture. Found clusters are then topics with a certain tag distribution. To compare topics, model cluster by a GMM. It will fail for the landscape dataset for instance.

## Text driven

Put photos in a graph where edge weight encode distance an run PLSA with network regularization.

## LGTA

R is a set of N Gaussian shaped region (weighted by α) from which topics are generated (instead of documents). Inside a region, topics are a multinomial distribution and inside a topic, words also. The log-likehood of the dataset given the parameters is optimized by EM. Prior knowledge of topic can be added. Tested over landscape, activity, Manhattan, festival, national park and car.
