function [f] = fullcost(A, K, cat1, cat2, P, Q)
    f = 0;
    tQ = A*Q;
    for k=1:K
        c1 = cat1 == k-1;        
        % cr1 = mean(P(1:2, c1), 2);
        c2 = cat2 == k-1;        
        % cr2 = mean(tQ(1:2, c2), 2);
        % f = f + (cr1-cr2)'*(cr1-cr2);
        f = f + (1/(sum(c1)*sum(c2)))*sum(sum(pdist2(P(1:2, c1)', tQ(1:2, c2)')));
    end
end
