function [ext] = getext(P)
    ext = diag([max(P(1,:)) - min(P(1,:)); max(P(2,:)) - min(P(2,:)); 1]);
end
