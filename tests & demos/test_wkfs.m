f2 = @(x) 0.1*x.^4 - 0.5 * x + 10;
noise_model = @(N) normrnd(0, 40, N, 1);

% sample x values from truncated Gamma distribution
gam = makedist('Gamma', 'a', 2, 'b', 1);
gam_trunc = truncate(gam, 0, 9);
data_dist = @(N) random(gam_trunc, N, 1);
N = 5000;
n_run = 1000;

rng(2); % set seed

reltol = 1e-6;

%% Generate data
% Sample data points
xs = data_dist(N);

% Evaluate model at data points
ys = f2(xs);

% Generate artificial measurements
zs = ys + noise_model(N);

%% Can the (W)KFs/KSs exactly recover OLS/WLS results?
% OLS/WLS
[ols_theta, ols_beta, ols_model, ols_Sigma, chisquared] = ols(xs, zs);
weights = 1./pdf(gam_trunc, xs);
[iwls_theta, iwls_beta, iwls_model, iwls_Sigma, iwls_chisquared] = wls(xs, zs, weights, false);
assert(all(diag(iwls_Sigma) > diag(ols_Sigma)));

% Now perform the same analyses using a Kalman filter / RTS smoother
% First, unweighted. Results should be exactly identical to OLS results.
Y = zs';
A = eye(2);
C = [1, xs(1)];
for i = 2:length(xs)
    C(:,:,i) = [1, xs(i)];
end
Q = zeros(2);
R = chisquared;
x0 = [0; 0];
P0 = [1e10, 0; 0, 1e10];
[params_filt, params_smooth, P_filt, P_smooth] = KF_RTS(Y, A, C, Q, R, x0, P0);
assert(max(abs((params_filt(:, end) - [ols_beta; ols_theta]) ./ [ols_beta; ols_theta]), [], 'all') < reltol);
assert(max(abs((params_smooth - [ols_beta; ols_theta]) ./ [ols_beta; ols_theta]), [], 'all') < reltol);
assert(max(abs((P_filt(:, :, end) - ols_Sigma) ./ ols_Sigma), [], 'all') < reltol);
for ii = 1:length(xs)
    assert(max(abs((P_smooth(:, :, ii) - ols_Sigma) ./ ols_Sigma), [], 'all') < reltol);
end

% And the same, but using the two-filter smoother
[params_filt, params_smooth, P_filt, P_smooth] = KF_TFS(Y, A, C, Q, R, x0, P0);
assert(max(abs((params_filt(:, end) - [ols_beta; ols_theta]) ./ [ols_beta; ols_theta]), [], 'all') < reltol);
assert(max(abs((params_smooth - [ols_beta; ols_theta]) ./ [ols_beta; ols_theta]), [], 'all') < reltol);
assert(max(abs((P_filt(:, :, end) - ols_Sigma) ./ ols_Sigma), [], 'all') < reltol);
for ii = 1:length(xs)
    assert(max(abs((P_smooth(:, :, ii) - ols_Sigma) ./ ols_Sigma), [], 'all') < reltol);
end

% Second, weighted. Results should be exactly identical to WLS results.
R = iwls_chisquared;
weights_normed = length(xs) * weights / sum(weights);
[params_filt_w, params_smooth_w, P_filt_w, P_smooth_w] = ...
    KF_RTS(Y, A, C, Q, R, x0, P0, 'sample_weights', weights_normed);
assert(max(abs((params_filt_w(:, end) - [iwls_beta; iwls_theta]) ./ [iwls_beta; iwls_theta]), [], 'all') < reltol);
assert(max(abs((params_smooth_w - [iwls_beta; iwls_theta]) ./ [iwls_beta; iwls_theta]), [], 'all') < reltol);
assert(max(abs((P_filt_w(:, :, end) - iwls_Sigma) ./ iwls_Sigma), [], 'all') < reltol);
for ii = 1:length(xs)
    assert(max(abs((P_smooth_w(:, :, ii) - iwls_Sigma) ./ iwls_Sigma), [], 'all') < reltol);
end

% And again the same, using the two-filter smoother
[params_filt_w, params_smooth_w, P_filt_w, P_smooth_w] = ...
    KF_TFS(Y, A, C, Q, R, x0, P0, 'sample_weights', weights_normed);
assert(max(abs((params_filt_w(:, end) - [iwls_beta; iwls_theta]) ./ [iwls_beta; iwls_theta]), [], 'all') < reltol);
assert(max(abs((params_smooth_w - [iwls_beta; iwls_theta]) ./ [iwls_beta; iwls_theta]), [], 'all') < reltol);
assert(max(abs((P_filt_w(:, :, end) - iwls_Sigma) ./ iwls_Sigma), [], 'all') < reltol);
for ii = 1:length(xs)
    assert(max(abs((P_smooth_w(:, :, ii) - iwls_Sigma) ./ iwls_Sigma), [], 'all') < reltol);
end

%% What happens if we attempt to estimate the noise covariances by minimizing the expected prediction errors?
% Note that the OLS/WLS model is false here (whence we're interested in weighted estimation!);
% therefore we cannot expect to recover the OLS/WLS noise covariance Q=0, R=chisquared. What do we
% get instead?

% First, _unweighted_ prediction error minimization and _unweighted_ Kalman filtering/smoothing
x0 = ones(4, 1);
opt_noise_params = fminunc(@(par) rwks_cost(par, zs', A, C), x0, ...
    optimoptions(@fminunc, 'Display', 'iter', 'Diagnostics', 'on', 'UseParallel', false, 'TypicalX', x0, ...
    'PlotFcn', 'optimplotx', 'MaxFunctionEvaluations', 1000));
Lopt = [opt_noise_params(1), opt_noise_params(2); 0, opt_noise_params(3)];
Qopt = Lopt'*Lopt;
Ropt = opt_noise_params(4)^2;
[~, params_smooth] = ...
    KF_TFS(Y, A, C, Qopt, Ropt);

% This should, again, exactly recover OLS estimates
assert(max(abs((params_smooth - [ols_beta; ols_theta]) ./ [ols_beta; ols_theta]), [], 'all') < reltol);

% Second, _weighted_ prediction error minimization and _weighted_ Kalman filtering/smoothing
opt_noise_params_w = fminunc(@(par) rwks_cost(par, zs', A, C, weights_normed), x0, ...
    optimoptions(@fminunc, 'Display', 'iter', 'Diagnostics', 'on', 'UseParallel', false, 'TypicalX', x0, ...
    'PlotFcn', 'optimplotx', 'MaxFunctionEvaluations', 1000));
Lopt_w = [opt_noise_params_w(1), opt_noise_params_w(2); 0, opt_noise_params_w(3)];
Qopt_w = Lopt_w'*Lopt_w;
Ropt_w = opt_noise_params_w(4)^2;
[~, params_smooth_w] = ...
    KF_TFS(Y, A, C, Qopt_w, Ropt_w, [], [], 'sample_weights', weights_normed);

% This should, again, exactly recover IWLS estimates
assert(max(abs((params_smooth_w - [iwls_beta; iwls_theta]) ./ [iwls_beta; iwls_theta]), [], 'all') < 2e-3);


%% plot results
ref_sigs = cell(1, 4);
ref_sigs{3} = [iwls_beta * ones(1, N); params_smooth(1, :); params_smooth_w(1, :)];
ref_sigs{4} = [iwls_theta * ones(1, N); params_smooth(2, :); params_smooth_w(2, :)];
ref_sig_labels = cell(1, 4);
ref_sig_labels{3} = {'OLS', 'WLS', 'PEM-optimal, unweighted RW-KS', 'IWPEM-optimal, weighted RW-KS'};
ref_sig_labels{4} = ref_sig_labels{3};
plot_signals([xs'; zs'; [ols_beta; ols_theta] * ones(1, N)], 1:N, {'x(t)', 'y(t)', 'offset', 'slope'}, 'k', ...
    'Time-varying regression under covariate shift', 'ref_sigs', ref_sigs, 'ref_sig_labels', ref_sig_labels);

% Plot results
figure;
scatter(xs, zs);
hold on;
x_grid = 0:.01:9;
plot(x_grid, f2(x_grid));
plot(x_grid, ols_model(x_grid));
plot(x_grid, iwls_model(x_grid));
legend('data', 'f(x)', 'OLS', 'IWLS');
xlabel('x');
ylabel('y=f(x)');

%% Function definitions

function cost = rwks_cost(par, ys, A, C, weights)
    L = [par(1), par(2); 0, par(3)];
    Q = L'*L;
    Rroot = par(4);
    R = Rroot^2;
    
    % We cannot calculate the gradients because C is time-varying.
    % Also, in the weighted case, we can only use the TFS.
    if nargin < 5  % unweighted
        
        [~, ~, ~, ~, ~, ~, innovation] = KF_RTS(ys, A, C, Q, R);
        cost = nansum(innovation.^2, 'all');
        if R > 0
            [~, ~, ~, ~, ~, ~, innovation_TFS] = KF_TFS(ys, A, C, Q, R);
            cost_TFS = nansum(innovation_TFS.^2, 'all');
            assert(abs((cost - cost_TFS) / cost) < 1e-4);
        end
    else  % weighted
        [~, ~, ~, ~, ~, ~, innovation] = ...
                    KF_TFS(ys, A, C, Q, R, [], [], 'sample_weights', weights);
        % (Weighted) sum of squared prediction errors
        cost =  nansum(weights(:) .* nansum(innovation.^2, 1)', 1);
    end
end