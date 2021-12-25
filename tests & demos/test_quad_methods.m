%% Test case 1: linear. 
% Mean and covariance approximation should be exact with all methods.
for ii = 1:1000
    % Progressively increasing dimension
    n = 1 + round(3*log10(ii));
    A = rand(n, n);
    func1 = @(x) A*x;
    mx = rand(n,1);
    % Progressively worse-conditioned cov matrices
    if n > 1
        eig_max_min_ratio_log = 8*ii/1000;
        covx = gen_rand_spd(n, eig_max_min_ratio_log);
    else
        covx = gen_rand_spd(1);
    end
    my_exact = A * mx;
    covy_exact = A * covx * A';
    [mu_ut, covy_ut] = sp_transform(func1, mx, covx, @sigma_points_classic);
    [mu_gh4, covy_gh4] = sp_transform(func1, mx, covx, @(m, c) gauss_hermite_n(m, c, 4));
    assert(max(abs(mu_ut - my_exact), [], 'all') < 1e-7);
    assert(max(abs(covy_ut - covy_exact), [], 'all') < 1e-7);
    assert(max(abs(mu_gh4 - my_exact), [], 'all') < 1e-7);
    assert(max(abs(covy_gh4 - covy_exact), [], 'all') < 1e-7);    
end

%% Test case 2: bilinear. 
% UT should only provide exact mean; GH-4 should provide exact mean and cov.
n = 3;
N = 1e8;
func2 = @(x) [x(1, :).*x(2, :); -5*x(3, :).*x(1, :); 0.1*x(3, :)];
for ii = 1:50
    mx = rand(n, 1);
    % Progressively worse-conditioned cov matrices
    eig_max_min_ratio_log = 8*ii/50;
    covx = gen_rand_spd(n, eig_max_min_ratio_log);
    % Monte Carlo approximation to exact solution
    X = mvnrnd(mx, covx, N)';
    assert(max(abs(mean(X, 2) - mx), [], 'all') < 1e-2);
    assert(max(abs(cov(X') - covx), [], 'all') < 1e-2);
    Y = func2(X);
    mu_mc = mean(Y, 2);
    covy_mc = cov(Y');
    
    [mu_ut, covy_ut] = sp_transform(func2, mx, covx, @sigma_points_classic);
    [~, max_abs_mu_err_idx] = max(abs(mu_ut - mu_mc));
    assert(abs((mu_ut(max_abs_mu_err_idx) - mu_mc(max_abs_mu_err_idx)) / mu_mc(max_abs_mu_err_idx)) < 1e-2);
    
    [mu_gh4, covy_gh4] = sp_transform(func2, mx, covx, @(m, c) gauss_hermite_n(m, c, 4));
    [~, max_abs_mu_err_idx] = max(abs(mu_gh4 - mu_mc));
    assert(abs((mu_gh4(max_abs_mu_err_idx) - mu_mc(max_abs_mu_err_idx)) / mu_mc(max_abs_mu_err_idx)) < 1e-2);
    [~, max_abs_cov_err_idx] = max(abs(covy_gh4(:) - covy_mc(:)));
    assert(abs((covy_gh4(max_abs_cov_err_idx) - covy_mc(max_abs_cov_err_idx)) / covy_mc(max_abs_cov_err_idx)) < 1e-2);
end
