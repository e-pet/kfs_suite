function [balcdata, cbhticks, cbhticklabels] = balance_cdata(cdata)
    % Compute empirical cumulative distribution function (ECDF) of the data
    nan_mask = isnan(cdata(:));
    pos_inf_mask = isinf(cdata(:)) & cdata(:) > 0;
    neg_inf_mask = isinf(cdata(:)) & cdata(:) < 0;
    normal_mask = ~nan_mask & ~pos_inf_mask & ~neg_inf_mask;
    
    [f_ecdf, x_ecdf] = ecdf(cdata(normal_mask));
    
    % evaluate inverse ECDF using linear interpolation
    [x_ecdf_unq1, idces1] = uniquetol(x_ecdf, 1e-7);
    [f_ecdf_unq2, idces2] = uniquetol(f_ecdf(idces1), 1e-7);
    x_ecdf_unq2 = x_ecdf_unq1(idces2);
    assert(length(x_ecdf_unq2) == length(f_ecdf_unq2));
    
    balcdata = zeros(size(cdata(:)));
    balcdata(normal_mask) = interp1(x_ecdf_unq2, f_ecdf_unq2, cdata(normal_mask), 'pchip');
    balcdata(nan_mask) = nan;
    balcdata(pos_inf_mask) = f_ecdf_unq2(end);
    balcdata(neg_inf_mask) = f_ecdf_unq2(1);
    balcdata = reshape(balcdata, size(cdata));
    
    % ticks + labels for the colorbar
    cbhticks = linspace(0, 1, 7) ; %Create 7 ticks from zero to 1
    cbhticklabels = num2cell(interp1(f_ecdf_unq2, x_ecdf_unq2, cbhticks, 'pchip'));
end