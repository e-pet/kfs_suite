function [theta, beta, fun, Sigma, chisquared, rsquared] = wls(xs, ys, weights, treat_weights_as_precision, fit_intercept)
%% WEIGHTED LEAST SQUARES estimation

    if nargin < 3 || isempty(weights)
        weights = ones(length(ys), 1);  % OLS!
    else
        assert(all(size(weights) == size(ys)));
    end
    
    if nargin < 4 || isempty(treat_weights_as_precision)
        treat_weights_as_precision = true;
    end
    
    if nargin < 5 || isempty(fit_intercept)
        fit_intercept = true;
    end

    if fit_intercept
        xs = [ones(size(xs, 1), 1), xs];
    end
    
    sqrt_weights = sqrt(weights);
    xsw = sqrt_weights.*xs;
    ysw = sqrt_weights.*ys;
    theta = xsw \ ysw;
    
    if nargout > 3     
        % See https://en.wikipedia.org/wiki/Ordinary_least_squares and
        % Faraway (2002): Practical Regression and Anova using R
        % for the calculation of the parameter estimate covariance Sigma.

        % estimation residual
        resw = ysw - xsw * theta;
        
        if nargin < 3
            % See https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
            chisquared = resw' * resw / (length(ys) - length(theta));       
        else
            % See https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
            % caution: I replaced n by sum(W) myself, because it seemed more 
            % reasonable to me... (the formula with n also does not pass my tests)
            %chisquared = resw' * resw / (sum(weights) - length(theta));
            chisquared = resw' * resw / (length(ys) - length(theta));
        end        
        
        % See my Zettelkasten note on WLS
        xsw2 = xsw' * xsw;   
        
        if treat_weights_as_precision
            % OLS or classical WLS     
            Sigma = chisquared * inv(xsw2);
        else
            % Non-classical case: W != inv(Sigma_y). For some reason we want to
            % weight individual measurements differently, but we still believe
            % all are measured with the same precision.
            % This will _always_ lead to higher parameter covariance.
            % (OLS/classical WLS are minimum-variance estimators...)
            Sigma = chisquared * ((xsw2 \ (xsw' * (weights .* xsw))) / xsw2);
        end
    end
    
    % Only do this now because the above code exploits that theta is the complete 
    % params vector, including the intercept (if it is estimated, that is).
    if fit_intercept
        beta = theta(1);
        theta = theta(2:end);
    else
        beta = 0;
    end    
    
    if nargout > 2
        fun = @(x) x * theta + beta;
    end
    
    if nargout > 5
        if fit_intercept
            ssres = sum((ys - fun(xs(:, 2:end))).^2);
        else
            ssres = sum((ys - fun(xs)).^2);
        end
        
        sstot = sum((ys - nanmean(ys)).^2);
        rsquared = 1 - ssres / sstot;
    end
end