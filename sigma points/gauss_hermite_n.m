function [sigma_points, weights] = gauss_hermite_n(mean, cov, degree_of_exactness)

    persistent sigma_points_base;
    persistent weights_base;
    persistent n;
    persistent num_points;
    
    % The normalized quadrature points + weights only need to be computed
    % once and should be reloaded during subsequent calls.
    nloc = length(mean);
    np_loc = round((degree_of_exactness + 1) / 2);
    if isempty(n) || ~(n == nloc && num_points == np_loc)
        n = nloc;
        num_points = np_loc;        
        fprintf('Computing %d-point Gauss-Hermite quadrature rule in %d dimensions.\n', num_points, n);
        fprintf('This means there are %d^%d=%d quadrature points in total.\n', num_points, n, num_points^n);
        % Compute the standard rule for mean = 0 and sigma^2 = 1,
        % corresponding to a scale factor b= 1/(2*sigma^2) = 1/2.
        [sigma_points_1D, weights_1D] = hermite_rule(num_points, 0, 0.5, 1);
        
        sigma_points_base = ndgridarr(n, sigma_points_1D)';
        weights_base = prod(ndgridarr(n, weights_1D), 2)';
        assert(abs(sum(weights_base) - 1) < 1e-8);
    end
    
    % Now scale and rotate the normalized points to fit on the given
    % distribution
    % For details, see Arasaratnam, Haykin, Elliott (2007):
    % Discrete-Time Nonlinear Filtering Algorithms Using
    % Gauss–Hermite Quadrature, Proc. IEEE
    R = robust_spd_chol(cov);
    sigma_points = R * sigma_points_base + mean;
    weights = weights_base;
end
