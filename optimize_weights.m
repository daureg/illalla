function res = optimize_weights(args)
    tic;
    A = args.A;
    disp(args.involved)
    disp(class(args.involved))
    involved = args.involved+1;
    notinvolved = args.notinvolved+1;
    nfold = size(involved, 1);
    [M, N] = size(A);
    Ain = -ones(1, N);
    bin = -16;
    lb = -bin/N/3*ones(1, N);
    ub = -3*bin/N*ones(1, N);
    thetas = zeros(nfold, N);
    distances = zeros(nfold, 2);
    allone = ones(N, 1) / sqrt(N);
    for i=1:nfold
        M = numel(notinvolved(i,:));
        x = lsqlin(A(notinvolved(i,:), :), zeros(M, 1), Ain, bin, [], [], lb, ub, [], optimoptions(@lsqlin,'Algorithm','active-set'));
        theta = x./norm(x);
        distances(i, :) = [norm(A(involved(i, :), :)*theta) norm(A(involved(i, :), :)*allone)];
        thetas(i, :) = theta;
    end
    toc
    res = struct('thetas', thetas, 'distances', distances);
end
