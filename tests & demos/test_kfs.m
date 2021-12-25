addpath('block-matrix-inverse-tools');
rng(1);

%% Linear test case: linear and nonlinear KFs should yield identical results
% Partially observed, stable (i.e. damped) oscillator, driven by strong
% white noise
A = [sqrt(3)/2.1,  1/2 ; -1/2, sqrt(3)/2.1];
Q = [2, 0; 0, 1];
C = [0, 1];
R = 3;
N = 1000;
x = zeros(2, N);
ym = zeros(1, N);

for ii = 2:N
    x(:, ii) = A * x(:, ii-1) + Q * randn(2, 1);
    ym(ii) = C * x(:, ii) + R * randn;
end

% These two should actually do exactly the same computation
[Xfiltered, Xsmoothed, Pfiltered, Psmoothed] = KF_RTS(ym, A, C, Q, R);
[Xfiltered2, Xsmoothed2, Pfiltered2, Psmoothed2] = KF_TFS(ym, A, C, Q, R);
[XfilteredNL1, XsmoothedNL1, ~, ~, n_iter1] = KF_RTS_NL(ym, A, C, Q, R);
assert(n_iter1 == 1)
assert(max(abs(Xfiltered - XfilteredNL1), [], 'all') < 1e-5);
assert(max(abs(Xsmoothed - XsmoothedNL1), [], 'all') < 1e-5);
assert(max(abs(Xfiltered2 - XfilteredNL1), [], 'all') < 1e-5);
assert(max(abs(Xsmoothed2 - XsmoothedNL1), [], 'all') < 1e-5);

% This performs a sigma-point linearization, which in the case of a
% nonlinear function should again result in (analytically) identical
% computations. Here, there might be a bit larger numerical difference,
% though.
Afunc = @(x) A * x;
[XfilteredNL2, XsmoothedNL2, ~, ~, n_iter2] = KF_RTS_NL(ym, Afunc, C, Q, R, [0;0]);
% Results in the second iteration should be exactly identical to results in
% the first iteration, and then the iterations should stop.
assert(n_iter2 == 2);
assert(max(abs((Xfiltered - XfilteredNL2) ./ Xfiltered), [], 'all') < 1e-5);
assert(max(abs((Xsmoothed - XsmoothedNL2) ./ Xsmoothed), [], 'all') < 1e-5);

% Generally, the smoother should perform better than the filter
assert(mean(abs(Xsmoothed - x), 'all') < mean(abs(Xfiltered - x), 'all'));
assert(mean(abs(Xsmoothed2 - x), 'all') < mean(abs(Xfiltered2 - x), 'all'));

% Uncertainty of RTSS and TFS should be identical
assert(all(abs((Psmoothed(:) - Psmoothed2(:)) ./ Psmoothed(:)) < 1e-4));

% Plot stuff
ref_sigs = cell(3, 1);
ref_sigs{1} = [Xfiltered(1, :); Xsmoothed(1, :)];
ref_sigs{2} = [Xfiltered(2, :); Xsmoothed(2, :)];
ref_sig_labels = cell(3, 1);
ref_sig_labels{1} = {'Data', 'Filter Estimate', 'Smoother Estimate'};
ref_sig_labels{2} = {'Data', 'Filter Estimate', 'Smoother Estimate'};
plot_signals([x; ym], 1:N, {'x1', 'x2', 'y'}, 'idx', 'Linear Test Case', ...
    'ref_sigs', ref_sigs, 'ref_sig_labels', ref_sig_labels);


%% Static LS regression: KS solution should be exactly equivalent to OLS
N = 100;
n = 2;
x = rand(N, 2);
sigma_e2 = 0.05;
ym = x(:, 1) - x(:, 2) + sigma_e2 * randn(N, 1);
theta_ols = x \ ym;

% KF solution to this problem
A = eye(2);
Q = zeros(2, 2);
C = zeros(1, 2, N);
for ii = 1:N
    C(:, :, ii) = x(ii, :);
end
R = sigma_e2;
[Xfiltered, Xsmoothed] = KF_RTS(ym', A, C, Q, R);
[Xfiltered2, Xsmoothed2] = KF_TFS(ym', A, C, Q, R);
assert(mean(abs(Xfiltered(:, end) - theta_ols)) < 1e-6);
assert(mean(abs(Xsmoothed(:, 1) - theta_ols)) < 1e-6);
assert(mean(abs(Xfiltered2(:, end) - theta_ols)) < 1e-6);
assert(mean(abs(Xsmoothed2(:, 1) - theta_ols)) < 1e-6);