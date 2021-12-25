function [sigma_points, weights] = sigma_points_classic(mean, cov, alpha)
    % Construct 2n+1 Sigma Points
    % alpha =1 results in classical sigma point transform; everything else results in the scaled unscented transform
    % (the latter is recommended).
    
    if nargin < 3
        alpha = 0.9;
    end
    
    n = length(mean);
    k = 3 / alpha / alpha - n;

    % Calculate the Sigma points
    sigma_points = zeros(n, 2 * n + 1);

    sigma_points(:, 1) = mean;
    cov_scale = (n + k) * cov;
    C = robust_spd_chol(cov_scale);
    
    for ii=2:n+1
        sigma_points(:, ii) = mean + alpha * C(:, ii-1);
    end
    
    for ii = n+2:2*n+1
        sigma_points(:, ii) = mean - alpha * C(:, ii - n - 1);
    end

    % Calculate the weight of each SP
    weights = zeros(1, 2 * n + 1);
    weights(1) = k / (n + k) / alpha / alpha + 1 - 1 / alpha / alpha;
    weights(2:end) = 1 / (2 * alpha * alpha * (n + k));

    assert(abs(sum(weights)-1) < 1e-8);
end