function R = blockmldivide(M, subsizes, b, skipcheck)
% BLOCKMLDIVIDE implement matrix left-divide (see @mldivide), as in the solution to M*x = b
% implemented as x = M \ b. Here, x = blockmldivide(M, subsizes, b) where 'subsizes' is defined as
% in @blockinv. See @blockinv for further details.

if nargin < 4 || ~skipcheck
    assert(ismatrix(M));
    assert(sum(subsizes) == size(M, 1) && size(M, 1) == size(M, 2));
    assert(sum(subsizes) == size(b, 1));
end

if length(subsizes) == 1
    % Base case - result is simply M \ b. Note: use built-in mldivide to get inverse, since
    % mldivide contains smart handling of matrices with special structure like diagonal or
    % upper-triangular matrices.
    R = M \ b;
else
    % Recursive case - apply formula from @blockinv with left-divide applied as we go. To recap, we
    % invert the input matrix M by dividing into sub-matrices [A B; C D], then computing and
    % concatenating sub-matrices [W X; Y Z]. The @mldivide result is, then, [[W X]*b; [Y Z]*b]
    
    % Extract sub-matrices A, B, C, D, b1, b2
    sub1 = subsizes(1);
    
    A = M(1:sub1, 1:sub1);
    B = M(1:sub1, sub1+1:end);
    C = M(sub1+1:end, 1:sub1);
    D = M(sub1+1:end, sub1+1:end);
    b1 = b(1:sub1, :);
    b2 = b(sub1+1:end, :);
    
    % Compute results Wb1+Xb2, Yb1+Zb2
    Aib1 = A \ b1;
    AiB = A \ B;
    CAiB = C * AiB;
    Zi = D - CAiB;
    
    Zb2 = blockmldivide(Zi, subsizes(2:end), b2, true);
    ZCAib1 = blockmldivide(Zi, subsizes(2:end), C * Aib1, true);
    Wb1Xb2 = Aib1 + AiB * ZCAib1 - AiB * Zb2;
    Yb1Zb2 = -ZCAib1 + Zb2;
    
    % Create output
    R = [Wb1Xb2; Yb1Zb2];
end
end