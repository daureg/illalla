# Geo Topic Model: Joint Modeling of User’s Activity Area and Interests for Location Recommendation

a method to recommend location to user based on topic modelling, user's daily life range and its topic interest. Here “document” are user history and “words” are locations, that are grouped by topic. More precisely, given the history of location of a user, predict which one will be next.

The main difference with LDA is that probability to of a location given the topic and user history is affected by its distance to the location in the history.

The prediction part performs better than baseline method but accuracy is quite low anyway. On the other hand, the topic extraction seems quite good.
