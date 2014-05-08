clear all; close all; clc;
c1 = [.3 1.2;.4 .3; .55 .68; .8 1.3];
% c1 = [.35 1.5;.4 .31; .55 .78; .8 .63; .9 1.1; .8 .8];
c2 = [.35 1.5;.4 .31; .55 .78; .8 .63; .9 1.1; .8 .8];
n = 32; m=ceil(2.3*n);
c1 = randn(n, 2);
c2 = randn(m, 2);
cl1=[0 0 1 1];
cl2=[0 0 0 1 1 1];
cl1 = rand(1, n) > .6;
cl2 = rand(1, m) > .6;
c1 = [normrnd(1, .3, n, 2); normrnd(-1, .3, n, 2)];
cl1 = [zeros(1, n) ones(1, n)];
c2 = [normrnd(-1, .3, m, 1) normrnd(1, .3, m, 1); normrnd(1, .3, m, 1) normrnd(-1, .3, m, 1)];
cl2 = [zeros(1, m) ones(1, m)];
% load('tsne_barcelona.mat')
% cl1 = cl;
% c1 = data;
% load('tsne_helsinki.mat')
% cl2 = cl;
% c2 = data;
% cl1 = not(cl2);
K=numel(unique(cl2));
% D=2;
% cr=zeros(K, D);
% for i=1:K;
%     cr(i,:) = mean(c2(cl1==i-1,:));
% end
P = [c1'; ones(1, numel(cl1))];
Q = [c2'; ones(1, numel(cl2))];
a = .9; b = .2; c = .2; d = .8; x = .1; y = .2;
A=[a b x; c d y; 0 0 1];
A(1:2, :) = randn(2, 3);
enlarge = max(0, getext(P)./getext(Q));
A = enlarge;
Aineq = zeros(2, 9);
Aineq(1, 1) = -1;
Aineq(2, 5) = -1;
bineq = [ -.65*enlarge(1, 1) -.65*enlarge(2, 2)];
Aineq = []; bineq = [];
Aeq = zeros(3, 9);
Aeq(1:3, 7:9) = eye(3);
% Aeq(4, [1 2 4 5]) = -1;
% beq = [0 0 1 -1]';
beq = [0 0 1]';
f = @(x)fullcost(reshape(x, 3, 3), K, cl1, cl2, P, Q);
options = optimoptions(@fmincon,'Algorithm', 'active-set', 'Display', 'iter', 'MaxIter', 1000, 'TolFun', 1e-8);
tic;
[x,fval,exitflag,output,lambda,grad,hessian] = fmincon(f, A(:), Aineq, bineq, Aeq, beq, [], [], [], options);
toc
% save('align', 'A');
nA=reshape(x, 3, 3)
pc(cl1, P, 'r')
hold on; pc(cl2, Q, 'b')
enlarge = max(0, getext(P)./getext(nA*Q));
hold on; pc(cl2, enlarge*nA*Q, 'g')
