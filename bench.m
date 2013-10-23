clear all; close all hidden; clc;
load('points.mat')
N = size(points, 1);
GREAT = 10000;
tic;
grav = mean(points);
tmp = points - repmat(grav, N, 1);
% TODO: get square distance ?
dst = sum(tmp.^2, 2);
toc
mean(dst)
std(dst)
tic;
dst = pdist(points);
toc
mean(dst)
std(dst)
pd = squareform(dst);
toc
pd = spdiags (GREAT*ones(size(pd, 1), 1), [0], pd);
toc
dst = min(pd, [], 2);
toc
mean(dst)
std(dst)
