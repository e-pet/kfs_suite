function [xopt, fopt] = latin_hypercube_search(costfun, lb, ub, npoints, fixed_vals, use_parallel)

    if nargin < 6
        use_parallel = false;
    end

    npar = length(lb);

    if nargin > 4 && ~isempty(fixed_vals)
        assert(iscell(fixed_vals));
        assert(length(fixed_vals) == npar);
        n_constraint = sum(~cellfun(@isempty, fixed_vals));
    else
        fixed_vals = cell(1, npar);
        n_constraint = 0;
    end
    
    points = zeros(npoints, npar);
    for ii = 1:npar
        if isempty(fixed_vals{ii})
            points(:, ii) = lb(ii) + lhsdesign(npoints, 1) * (ub(ii) - lb(ii));
        else
            points(:, ii) = fixed_vals{ii};
        end
    end

    xopt = ones(size(ub));
    fopt = inf;
    
    disp(['Running latin hypercube search, n=', num2str(npoints), ' sample points in ', num2str(length(lb)), '-D.']);
    disp(['Values along ', num2str(n_constraint), ' dimensions are fixed.']);
    reverseStr = '';
    if use_parallel
        fvals = zeros(1, npoints);
        parfor ii = 1:npoints
            fvals(ii) = costfun(points(ii, :));
        end
        [fopt, idx] = min(fvals);
        xopt = points(idx, :);
    else
        for ii = 1:npoints
            fii = costfun(points(ii, :));
            if fii < fopt
                fopt = fii;
                xopt = points(ii, :);
            end
           % Display the progress
           percentDone = 100 * ii / npoints;
           msg = sprintf('Percent done: %3.1f\n', percentDone); %Don't forget this semicolon
           fprintf([reverseStr, msg]);
           reverseStr = repmat(sprintf('\b'), 1, length(msg));        
        end        
    end
end