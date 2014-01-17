# Mapping the World’s Photos

Analyse interplay between content (tags and image features) and structure (geo info). Additional structure could be social link between photographers. The goal is to predict location given other information (at city scale and landmark scale) and return representative image of a given location.

## Mean shift

non parametrized estimation of mode of an unknown distribution given sample and approximate scale. Iteratively estimate the gradient to find its zeroes (i.e. start from a seeding location and follow the gradient until a local maxima).

## Classification

create a visual vocabulary using SIFT vectors as input. Tags form a vector space. Each method outperform baseline and they are even better when combined. Adding time information (by examining photos of the same user in a 30 minutes window) also increases accuracy.

## Selecting representative image as a graph problem

[results](http://www.cs.cornell.edu/~crandall/photomap/)
