addpath('cbrewer');

%% First, check that the likelihood values are actually correct for very few data points
A=1;C=1;Q=1;R=1;ys=1.2;x0=1;P0=1;
[~, ~, ~, ~, ~, ~, ~, neg_log_likelihood] = KF_RTS(ys, A, C, Q, R, x0, P0);
% Likelihood of y
y_cov = C*P0*C'+R;
assert(abs((neg_log_likelihood - 0.5 * log(det(y_cov)) - 0.5*(ys-x0)'/y_cov*(ys-x0)-0.5*log(2*pi)) / neg_log_likelihood) < 1e-6);

[~, ~, ~, ~, ~, ~, ~, neg_log_likelihood] = KF_TFS(ys, A, C, Q, R, x0, P0);
% Likelihood of y
y_cov = C*P0*C'+R;
assert(abs((neg_log_likelihood - 0.5 * log(det(y_cov)) - 0.5*(ys-x0)'/y_cov*(ys-x0)-0.5*log(2*pi)) / neg_log_likelihood) < 1e-6);


%% Set up example
% estimate a time-varying "constant" and learn both noise matrices from the data by ML maximization
N = 5000;
meas_noise_std = 0.8;
proc_noise_std = 0.3;
A = 1;
C = 1;
x = zeros(1, N+1);
x(1) = 0;
y = zeros(1, N);
rng(3); % set seed
for ii = 1:N
    x(ii+1) = A * x(ii) + proc_noise_std * randn;
    y(ii) = C * x(ii+1) + meas_noise_std * randn;
end

plot_signals([x(2:end); y], 1:N, {'x', 'y'}, 'sample');


%% What do the cost function and gradients look like?
max_abs_grad_diff_q = zeros(1, 4);
max_abs_grad_diff_r = zeros(1, 4);
avg_rel_grad_diff_q = zeros(1, 4);
avg_rel_grad_diff_r = zeros(1, 4);
kk = 1;
stepsizes = [0.2, 0.1, 0.05, 0.025];
for stepsize = stepsizes
    [qs, rs] = meshgrid(-0.7:stepsize:0.7, -0.4:stepsize:1.5);
    fs = zeros(size(qs));
    gradq_norm = zeros(size(qs));
    gradr_norm = zeros(size(rs));
    gradq = zeros(size(qs));
    gradr = zeros(size(rs));
    for ii=1:size(qs, 1)
        for jj = 1:size(qs, 2)
            [fs(ii, jj), grad] = rwks_cost([qs(ii, jj), rs(ii, jj)], y, A, C);
            gradq(ii, jj) = grad(1);
            gradr(ii, jj) = grad(2);  
            % normalize gradients for plotting
            gradq_norm(ii, jj) = grad(1) / sqrt(grad(1)^2 + grad(2)^2);
            gradr_norm(ii, jj) = grad(2) / sqrt(grad(1)^2 + grad(2)^2);
        end
    end
    
    % for gradient evaluation, consider only a quadrant with non-zero variances, because near zero
    % the log likelihood diverges, and hence finite difference gradient approximations go haywire
    cols = qs(1, :) > 0.05;
    rows = rs(:, 1) > 0.05;
    
    % calculate numerical gradient approximation by finite differences
    [num_grad_q, num_grad_r] = gradient(fs(rows, cols), stepsize);
    
    % only look at inner nodes at which central differences are used
    num_grad_q_inner = num_grad_q(:, 2:end-1);
    num_grad_r_inner = num_grad_r(2:end-1, :);
    grad_q_inner = gradq(rows, (find(cols, 1)+1):(find(cols, 1, 'last')-1));
    grad_r_inner = gradr((find(rows, 1)+1):(find(rows, 1, 'last')-1), cols);
    
    % calculate errors in the gradients, both absolute and relative
    abs_grad_diff_q = abs(num_grad_q_inner - grad_q_inner);
    abs_grad_diff_r = abs(num_grad_r_inner - grad_r_inner);
    rel_grad_diff_q = abs((num_grad_q_inner - grad_q_inner) ./ grad_q_inner);
    rel_grad_diff_r = abs((num_grad_r_inner - grad_r_inner) ./ grad_r_inner);
    max_abs_grad_diff_q(kk) = max(abs_grad_diff_q, [], 'all');
    max_abs_grad_diff_r(kk) = max(abs_grad_diff_r, [], 'all');
    avg_rel_grad_diff_q(kk) = mean(rel_grad_diff_q, 'all');
    avg_rel_grad_diff_r(kk) = mean(rel_grad_diff_r, 'all');
    kk = kk + 1;
end
figure;
subplot(1, 2, 1);
semilogx(stepsizes, max_abs_grad_diff_q)
hold on;
plot(stepsizes, max_abs_grad_diff_r)
xlabel('Finite difference step size');
ylabel('Absolute gradient error');
legend('Max abs error in df/dq', 'Max abs error in df/dr');

subplot(1, 2, 2);
semilogx(stepsizes, avg_rel_grad_diff_q)
hold on;
plot(stepsizes, avg_rel_grad_diff_r)
xlabel('Finite difference step size');
ylabel('Relative gradient error');
legend('Avg rel error in df/dq', 'Avg rel error in df/dr');

figure;
[balcdata, cbhticks, cbhticklabels] = balance_cdata(fs);
contourf(qs, rs, balcdata);
colormap(gca, jet(11));
cbh = colorbar;
cbh.Ticks = cbhticks;
cbh.TickLabels = cbhticklabels;
xlabel('std(q)');
ylabel('std(r)');
cbh.Label.String = '-log p(y)]';
hold on;
quiver(qs, rs, gradq_norm/2, gradr_norm/2);


%% estimate time-varying parameters using standard RW-KS
% Use ML optimization to find noise covariances
% first, perform latin hypercube search @ 50 points to find a good starting point (ML optimization is nonconvex)
lb = [0, 0];
ub = [5, 5];
par0 = latin_hypercube_search(@(par) rwks_cost(par, y, A, C), lb, ub, 50);
disp(['Starting point selected by LHS is: ', num2str(par0)]);

typicalx = par0;
opt_noise_params = fminunc(@(par) rwks_cost(par, y, A, C), par0, ...
    optimoptions(@fminunc, 'Display','iter', 'Diagnostics', 'on', 'UseParallel', true, 'TypicalX', typicalx, ...
    'PlotFcn', 'optimplotx', 'SpecifyObjectiveGradient', true, ...
    'Algorithm', 'trust-region', 'FunctionTolerance', 1e-15, 'OptimalityTolerance', 1e-10, 'StepTolerance', 1e-10));
Qopt = opt_noise_params(1)^2;
Ropt = opt_noise_params(2)^2;

% Do the estimated noised covariances agree with the true covariances used for the simulation?
% (The bounds can be tightened by simulating more than 5k samples, but it really takes _a lot_ of
% samples to get really accurate.)
assert(abs((Qopt - proc_noise_std^2) / Qopt) < 1e-2); 
assert(abs((Ropt - meas_noise_std^2) / Ropt) < 2e-2);

[~, params_smoothed_rwks_opt] = KF_RTS(y, A, C, Qopt, Ropt);

%% plot results
ref_sigs = cell(1, 2);
ref_sigs{1} = [params_smoothed_rwks_opt(1, :)];
ref_sig_labels = cell(1, 2);
ref_sig_labels{1} = {'Truth', 'ML-optimal RW-KS'};
plot_signals([x(2:end); y], 1:N, {'x(t)', 'y(t)'}, 't', ...
    'ML-optimal RTS', 'ref_sigs', ref_sigs, 'ref_sig_labels', ref_sig_labels);


%% Function definitions

function [cost, grad] = rwks_cost(par, ys, A, C)
    Qroot = par(1);
    Q = Qroot^2;
    Rroot = par(2);
    R = Rroot^2;
    
    if nargout > 1
        % get gradient of the cost as well  
        [~, ~, ~, ~, ~, ~, ~, neg_log_likelihood, dEnergydQroot, dEnergydRroot] = ...
            KF_RTS(ys, A, C, Q, R, [], [], 'Qroot', Qroot, 'Rroot', Rroot);
        grad = [dEnergydQroot, dEnergydRroot];
    else
        [~, ~, ~, ~, ~, ~, ~, neg_log_likelihood] = KF_RTS(ys, A, C, Q, R);
    end
    cost = neg_log_likelihood;
    if R > 0
        [~, ~, ~, ~, ~, ~, ~, neg_log_likelihood_TFS] = ...
                KF_TFS(ys, A, C, Q, R);
        assert(abs((neg_log_likelihood - neg_log_likelihood_TFS) / neg_log_likelihood) < 1e-3);
    end
end