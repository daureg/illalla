clear all; close all;
load('../mfreq/freq_200__background.mat')
dc=sort(c(c>0), 'descend');
load('../mfreq/freq_100__background.mat')
cc=sort(c(c>0), 'descend');
load('../mfreq/freq_40__background.mat')
qc=sort(c(c>0), 'descend');
% loglog(1:length (dc), dc, 1:length (cc), cc, 1:length (qc), qc)
% xlabel 'rank of cell'; ylabel 'photo count per cell'; title 'Photos distribution';
% legend('200 grid', '100 grid', '40 grid');

config.filename = 'photos_distrib.tex';
config.standalone = 1;
config.runtex = 0;
config.axistype = 'loglogaxis';
config.xlabel = 'rank of cell';
config.ylabel = 'photo count per cell';
genes={'200 grid', '100 grid', '40 grid'};
cols ={'FireBrick', 'LimeGreen', 'DodgerBlue'};
p.smooth = 1;
p.x = [1:600 601:18:length(dc)];
p.y = dc(p.x);
p.color = cols{1};
p.legend = genes{1};
P{1} = p;

p.x = [1:600 601:6:length(cc)];
p.y = cc(p.x);
p.color = cols{2};
p.legend = genes{2};
P{2} = p;

p.x = 1:length(qc);
p.y = qc;
p.color = cols{3};
p.legend = genes{3};
P{3} = p;
printpgf(config,P);
