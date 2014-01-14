tags are unstructured but there is an underlying semantic

define place semantic or query's dominant location (wang 05): Place tags exhibit spatial usage patterns that are significantly geographically localized.

Ratt 07: same method in time

tags pattern are auto correlated but only below a certain scale: above, it collapses into a single burst

first order (using only tag at a time) vs second order co occurence

Generics steps:

1. scale K=(k1, …, kn) from the data or not

2. region R=(ri) rectangular or more complicated

3. statistics computation over all region

4. aggregate to assess significance

5. link tag with relevant regions

Baseline methods:

- Naive scan, find tag frequency that differs a lot from the average

- spatial scan, discrepancy above a threshold

- TagMaps TF-IDF: k-mean (euclidean distance) photo cluster defines regions. Inside each region, compute tf-idf score of every tags + user score to avoid flood. Then look for tag appearing with a high score in a single region.

- scale structure identification: At scale k form a graph of tag position where edge if closer than d_k. Regions are then connected component. Compute the entropy of the size of these regions E_k,x. Then either find low entropy tag over all size. Or you can walk trough every scale and wait until entropy reach 0 (all tags in one region), which should happen fast for place tag.

Weighted average of normalized score of other methods

Manual ground truth: for each tag, determine if it's a place and if so, where.

Errors come either from bias (from lack of data) or for some methods, unusual shape.

Related work:

- equivalence with time detection

- MAUP, normalize region (by population) or do it multi-scale

- smoothing data (Brunsdon 02): Geographically weighted regression

- Geographic Information Retrieval

- textual approach to extract information from tagging system
