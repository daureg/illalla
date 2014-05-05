function pc(cl, pts, col)
    for k=1:numel(unique(cl))
        if (k==1)
            mark = '+';
        else
            mark = '*';
        end
        plot(pts(1, cl==k-1), pts(2, cl==k-1), strcat(col, mark));
        hold on
    end
end
