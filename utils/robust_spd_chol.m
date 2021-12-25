function R = robust_spd_chol(C)
% ROBUST_SPD_CHOL Compute cholesky decomposition and attempt to recover if
% C is not SPD (but close to it)
%
% In most cases, this is equivalent to calling chol directly.
% If chol fails for some reason, the first recovery attempt is to add a
% small identity matrix as a means of regularization, and run chol on this.
% If that still fails, eigs are computed, small negative eigs modified to
% positive values, and the "SPDed" matrix recomputed. Chol is then run on
% that matrix. If that _still_ fails, there's nothing else we can do.
% An error is thrown in that case.
%
    [R, flag] = chol(C, 'lower');
    if flag ~= 0
        % We're in trouble; the covariance matrix isn't positive definite
        % First attempt to save ourselves by regularization:
        C_reg = C + 1e-2*min(abs(C(:))) * eye(size(C, 1));
        [R, flag] = chol(C_reg, 'lower');
        if flag ~= 0
           % Still not working; last attempt: fiddle with eigs
            [V, D] = eig(C);

            % set any of the eigenvalues that are <= 0 to some small positive value
            for n = 1:size(D,1)
                if (D(n, n) <= 1e-10)
                    D(n, n) = 1e-6;
                end
            end
            % recompose the covariance matrix, now it should be positive definite.
            C_spd = V*D*V';

            [R, flag] = chol(C_spd, 'lower');
            if flag ~= 0
                error('Error: covariance matrix is not SPD, and SPDness cannot be recovered.');
            end
        end
    end
end