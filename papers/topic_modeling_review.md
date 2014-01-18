# Introduction to Probabilistic Topic Models

Topic modeling algorithms are unsupervised statistical methods that analyze the words of the original texts to discover the themes that run through them, how those themes are connected to each other, and how they change over time.

# LDA

topics are distribution over fixed vocabulary. Fixing a distribution over topics, document are generated words by words.

Words are observed variable but the topics, their membership and topics proportion are hidden so we need to compute the posterior. It's intractable so it's approximated either by sampling (Markov Chain) or optimizing a variational distribution in the sense of KL divergence.

# Extensions

- don't assume bag of words (doesn't matter for tags)

- order of documents over time: dynamic topic model (5)

- variable number of topic and hierarchy

- incorporating metadata such as author (author similarity based on their topic proportions)

- this mixed-membership model can be applied to non textual data as well (mixture model?)
