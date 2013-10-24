clear all; close all hidden; clc;
N = 1000
weigths = [.4 .3 .3];
params = [37.8 -122.47 .05/3;
	  37.75 -122.44 .04/3;
	  37.68 -122.42 .03/3];
[~, idx]=max(rand(N, 1) < cumsum(weigths), [], 2);
points = zeros(N, 2);
candidates = zeros(N, 6);
for i = 0:numel(weigths)-1
	candidates(:, [2*i+1 2*i+2]) = [normrnd(params(i+1, 1), params(i+1, 3), N, 1) normrnd(params(i+1, 2), params(i+1, 3), N, 1) ];
end
bounds = int32(N*cumsum (weigths ));
for i = 0:numel(weigths)-1
	upper = bounds(i+1);
	if (i == 0)
		lower = 1;
	else
		lower = bounds(i)+1;
	end
	points(lower:upper, :) = candidates(lower:upper, [2*i+1 2*i+2]);
end
save('-v7', 'points.mat', 'points')
