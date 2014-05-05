function [H] = get_hessian(kl1, kl2, Q)
    H = zeros(9, 9);
    for k=1:numel(unique(kl1))
        c1 = kl1 == k-1;        
        c2 = kl2 == k-1;        
        rel = Q(:, c2);
        % sum all submatrices obtained from q'*q
        qq = reshape(sum(im2col(rel(:)*rel(:)', [3 3], 'distinct'), 2), 3, 3);
        H = H + sum(c1)*kron(qq, eye(3));
    end
    H = -2*H
end
