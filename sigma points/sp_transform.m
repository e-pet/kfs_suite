function [res_mean, res_cov, cross_cov] = sp_transform(func, x_marginal, P_marginal, sp_scheme)

    if nargin < 4
        sp_scheme = @sigma_points_classic;
    end

    % Calculate Sigma Points
    [sigma_points, weights] = sp_scheme(x_marginal, P_marginal);
    sqrt_weights = sqrt(weights);
    
    % Send the sigma points through func
    sigma_points_transformed = func(sigma_points);
    assert(size(sigma_points_transformed, 2) == size(sigma_points, 2));
    
    % transform mean
    res_mean = sum(weights .* sigma_points_transformed, 2);

    % transform variance
    sp_transf_diff_weighted = sqrt_weights .* (sigma_points_transformed - res_mean);
    res_cov = sp_transf_diff_weighted * sp_transf_diff_weighted';
    
    % calculate cross-covariance
    sp_diff_weighted = sqrt_weights .* (sigma_points - x_marginal);
    cross_cov = sp_diff_weighted * sp_transf_diff_weighted';
    
    if size(res_cov, 1) == 1 && ~isnan(res_cov)
        assert(res_cov >= 0);
    end
end