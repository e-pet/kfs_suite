function [params, offset, fun, Sigma, chisquared, rsquared] = ols(X, y, fit_intercept)
    % Ordinary least squares (OLS) assumes errors only on y. 
    if nargin < 3 || isempty(fit_intercept)
        fit_intercept = true;
    end
    
    [params, offset, fun, Sigma, chisquared, rsquared] = wls(X, y, [], [], fit_intercept);
end