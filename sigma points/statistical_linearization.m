function [M, offset_mean, offset_cov] = statistical_linearization(func, marginal_mean, marginal_cov, sp_scheme)

    if nargin < 4
        sp_scheme = @sigma_points_classic;
    end
    
    [res_mean, res_cov, cross_cov] = sp_transform(func, marginal_mean, marginal_cov, sp_scheme);
    
    % See Herzog, Petersen, Rostalski (2019): 
    % Iterative Approximate Nonlinear Inference via
    % Gaussian Message Passing on Factor Graphs
    M = cross_cov' / marginal_cov;
    offset_mean = res_mean - M * marginal_mean;
    
    % The difference in the following formula can easily lead to negative
    % covariances due to numerical errors. In 1-D this is easy to fix. In
    % N-D it is more complex (and expensive!), and we do not attempt to 
    % detect or fix it.
    if size(res_cov, 1) == 1
        offset_cov = max(res_cov - M * marginal_cov * M', 0);
    else
        offset_cov = res_cov - M * marginal_cov * M';
    end
end