function spd = gen_rand_spd(n, eig_span_log)
    % Generates random symmetric positive definite nxn matrix with maximum entry 1.
    % If eig_span_log is given, log10(eigmax/eigmin) = eig_span_log is enforced.
	%
	% Eike Petersen, 2021
	
    if nargin < 2
        % Credit for this simple version goes to Walter Roberson, 
        % https://de.mathworks.com/matlabcentral/answers/417385-how-can-i-generate-random-invertible-symmetric-positive-semidefinite-square-matrix-using-matlab#answer_335343
        while true
            A = rand(n, n);
            if rank(A) == n; break; end    % will be true nearly all the time
        end
        spd = A' * A;
        
    else
        if n == 1
            error('There is only one eigenvalue if n == 1! Cannot enforce eig_span_log.');
        end
        eig_min = 10^(-eig_span_log);
        eig_max = 1;
        if n > 2
            other_eigs = eig_min + (1 - eig_min) * rand(1, n-2);
        else
            other_eigs = [];
        end
        D = diag([eig_max, sort(other_eigs), eig_min]);
        % To ensure that eigenvalues stay unchanged and the result is SPD, sample a random orthonormal eigen basis
        U = RandOrthMat(n);
        spd = U*D*U';
    end
    
    % normalize to max value 1
    spd = spd / max(spd(:));
end