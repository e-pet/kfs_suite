matSize = 100;
blocksizes = 10 * ones(1, 10);
nTests = 5;
tol = 1e-7;

for i=1:nTests
    M = randn(matSize);
    b = randn(matSize, 1);
    
    % Get built-in results
    Mi = inv(M);
    Mib = M \ b;
    bMi = b' / M;
    
    % Get custom results
    my_Mi = blockinv(M, blocksizes);
    my_Mib = blockmldivide(M, blocksizes, b);
    my_bMi = blockmrdivide(M, blocksizes, b');
    
    % Assert match
    assert(all(abs(Mi(:) - my_Mi(:)) < tol));
    assert(all(abs(Mib(:) - my_Mib(:)) < tol));
    assert(all(abs(bMi(:) - my_bMi(:)) < tol));
end