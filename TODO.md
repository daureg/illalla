1. Can we also try with Paris-Barcelona? First we know Barcelona better than SF, and second European cities are more likely to be similar.
1. I wanted to do Barcelona at some point today but then I did something else and unfortunately, I need physical access to the computer to launch the script so I'll have to wait until tomorrow
1. Good.


2. Can you add a location that includes a park or somewhere that people go on weekend (local people, not tourists)? I think that this is more distinct than what we have already, and it may be instructive to see how it performs.
2. Yes, I could ask some friends about that in Paris at least. I was also considering adding another kind of neighborhood, that is more “objective”: those where real estate is expensive. But I couldn't find a map of price in Barcelona; maybe you know such places?
2. The area bounded by < Placa Catalunya -- Passeig de Gracia -- Rambla de Catalunya (not 'La Rambla') -- Diagonal > should be quite expensive. Also Sarria-Sant Gervasi in the north west.

 Le jardin du luxembourg, dit luco;
les quais !!  Dans le 13 e vers le batofar et le wanderlust
ou sur le canal st martin très hypster le lieu à pique nique
les tuileries à la limite
Buttes chaumont
Monceau
Plage *éphémère* pont alexandre 3. Quais derrière notre dame et vers jussieu

3. What is the definition of JSD when the probability distribution is on a vector space and not on 1-D? I think that you have mentioned, but I do not recall...
3. I haven't tell because I'm not quite sure. Right now, I'm just computing JSD over each dimension independently and then combining the results (which I think make some sense because not all the dimensions are comparable: like the number of visitors and frequency of check-ins in the morning). But in the definition, this is just integrating probability density function so the dimension should not matter. Although in practice, I'm not convinced we can accurately estimate the pdf in 30 dimensions from around 70 points. I'll take a look at “kNN-based high-dimensional Kullback-Leibler distance for tracking”…
3. Yes, it is reasonable, but it may have some problems. Let's keep thinking about it and try different things.

4. Can we also try the same experiment using some "baseline" distance measures? By baseline I mean conceptually simpler than EMD and JSD. In particular:
4a. For a neighborhood just compute the average vector, and then for two neighborhoods compute the distance between their averages.
4b. For a neighborhood cluster all vectors in k clusters, and then for two neighborhoods compute the distance as the min-cost matching of their cluster centers (so for k=1 it is equivalent to 4a). For computing the min-cost matching there should be easy to find code.
4. I will try that as well.
4. Good.

5. (the other way: do something more complex, instead of doing something simpler). I think that there is a version of EMD that it does not require to move the whole mass but only a fraction of it, say 75%. This is a meaningful measure for our
5. From what I have read, if the two regions don't have the same mass, the algorithm moves all the mass from the smallest one to cover as much as possible of the largest one. So if I multiply all the weight in the target city by k > 1, there would be some left over that can be considered as outliers.
5. Not quite. With the modification you are suggesting the distribution with the less mass will always be matched completely. I was thinking the variant that some mass is left unmatched in both distributions. Perhaps there is a transformation that can be done to the distributions that allows to use the existing implementation of EMD.
