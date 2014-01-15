“GeoScope is a data streams solution that detects correlations between topics and locations in a sliding window, in addition to analyzing topics and locations independently”.

φ, the dominance of topic tx in location li and ψ, the support of location li for topic tx, are favored statistical significance like χ². To avoid unpopular correlation, report only the locations that are at least θ-frequent.

Finding an exact solution to this problem requires keeping track of every possible pair, which is intractable. GeoScope is thus an approximation, made by limiting the number of region tracked and the of tracked topic within

The algorithm is based on based on sketch data structure. Memory and time complexity are sub linear. Proof of guaranteed accuracy

GeoMap tool for Google Chart
