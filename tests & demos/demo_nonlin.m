addpath('block-matrix-inverse-tools');

%% Nonlinear demo
% Stable (i.e. damped) oscillator, driven by strong white noise, with a nonlinear measurement function.
rng(1);
A = [sqrt(3)/2.1,  1/2 ; -1/2, sqrt(3)/2.1];
Q = [2, 0; 0, 1];
R = 1;
N = 1000;
x = zeros(2, N);
ym = zeros(1, N);
C = @(x) 0.1*x(1, :).^2 + 5*sin(x(2, :)/10);

for ii = 2:N
    x(:, ii) = A * x(:, ii-1) + Q * randn(2, 1);
    ym(ii) = C(x(:, ii)) + R' .* randn(1, 1);
end

% Unscented transform
[Xfiltered_UT, Xsmoothed_UT] = KF_RTS_NL(ym, A, C, Q, R, 'return_iter', true);
% Gauss-Hermite rule of exactness order 2
[Xfiltered_GH2, Xsmoothed_GH2] = KF_RTS_NL(ym, A, C, Q, R, ...
    'quad_rule', @(mean, cov) gauss_hermite_n(mean, cov, 4), 'return_iter', true);
% Gauss-Hermite rule of exactness order 4
[Xfiltered_GH4, Xsmoothed_GH4] = KF_RTS_NL(ym, A, C, Q, R, ...
    'quad_rule', @(mean, cov) gauss_hermite_n(mean, cov, 8), 'return_iter', true);

% How does the MSE evolve as a function of iterations?
mse_UT = zeros(length(Xsmoothed_UT), 1);
for ii = 1:length(Xsmoothed_UT)
    mse_UT(ii) = mean(abs(Xsmoothed_UT{ii}(1:2, :) - theta), 'all');
end
mse_GH2 = zeros(length(Xsmoothed_UT), 1);
for ii = 1:length(Xsmoothed_GH2)
    mse_GH2(ii) = mean(abs(Xsmoothed_GH2{ii}(1:2, :) - theta), 'all');
end
mse_GH4 = zeros(length(Xsmoothed_UT), 1);
for ii = 1:length(Xsmoothed_GH4)
    mse_GH4(ii) = mean(abs(Xsmoothed_GH4{ii}(1:2, :) - theta), 'all');
end

figure('name', 'MSE per iteration', 'NumberTitle', 'off');
tiledlayout(1, 3);
h1 = nexttile;
plot(mse_UT);
xlabel('Iteration');
ylabel('MSE');
title('Unscented transform');

h2 = nexttile;
plot(mse_GH2);
xlabel('Iteration');
ylabel('MSE');
title('3-point Gauss-Hermite rule');

h3 = nexttile;
plot(mse_GH4);
xlabel('Iteration');
ylabel('MSE');
title('7-point Gauss-Hermite rule');
linkaxes([h1 h2 h3]); 

% Plot stuff
ref_sigs = cell(3, 1);
ref_sigs{1} = [Xsmoothed_GH2{end}(1, :)];
ref_sigs{2} = [Xsmoothed_GH2{end}(2, :)];
ref_sig_labels = cell(3, 1);
ref_sig_labels{1} = {'Data', 'GH4-RTS'};
ref_sig_labels{2} = {'Data', 'GH4-RTS'};
plot_signals([x; ym], 1:N, {'x1', 'x2', 'y'}, 'idx', 'Nonlinear demo', ...
    'ref_sigs', ref_sigs, 'ref_sig_labels', ref_sig_labels);