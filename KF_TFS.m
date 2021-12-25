function [X_filt, X_smooth, P_filt, P_smooth, MeasNoiseFilt, MeasNoiseSmoothed, innovation, energy] = ...
    KF_TFS(Y, A, C, Q, R, varargin)
% KF_TFS sequential Kalman Filter and Two-Filter Smoother
% Linear Kalman Filter and Smoother; sequential implementation of Joseph
% stabilized, symmetrized form. Only works for diagonal measurement noise
% covariance matrices R!
%
% To use a time-varying system description, pass 3d arrays for A, C, and/or Q,
% where the last index iterates over sample indices and/or an array for R,
% where the rows indicate different samples.
%   x(k+1) = A(k) * x(k) + q
%   y(k) = C(k) * x(k) + r
% Inputs are currently not considered.
%
% For details on the KF and its sequential implementation and the TFS see Dan Simon, Optimal State Estimation (2006), p. 150.
% The formulation of the energy function calculation is due to Simo Särkkä, "Bayesian filtering and smoothing".
% The formulation of state constraints is due to section 2.4 of "Kalman filtering with state constraints: a survey of linear and
% nonlinear algorithms" by Dan Simon, IET Control Theory and Applications, 2009.
% For an explanation of what the weighted version of this does (and why you might want it), see
% Eike Petersen, "Model-based Probabilistic Inference for Monitoring Respiratory Effort using the Surface Electromyogram"
% (Dissertation, forthcoming).
% Some of the implementation details are my own. (Eike Petersen)
%
%   INPUT:  Y           ->  rxN matrix where r is the number of measurement and
%                           N is the length of the measured signals.
%           A           ->  System matrix.
%           C           ->  Observation matrix
%           Q           ->  Process noise covariance.
%                           Should be either of the following:
%                           - mxm, specifying a constant matrix.
%                           - mxmxN, specifying a time-varying matrix.
%                           - 1xm, specifying the diagonal of a constant
%                           matrix.
%                           - Nxm, specifying a time-varying diagonal
%                           matrix.
%           R           ->  1xr vector with oberservation noise covariance
%                           matrix diagonal entries. If time-varying, this
%                           should be Nxr.
%           x0          ->  Vector of length m with initial state.
%           P0          ->  mxm matrix with initial state covariance.
%           handle_nans ->  handle 'nan' values in measurements or
%                           observation matrices by simply treating this
%                           measurement as completely unknown (infinite 
%                           measurement variance). Default is true.
%           xmin        ->  Lower bound for state values. (Optional)
%           xmax        ->  Upper bound for state values. (Optional)
%           sample_weights -> Weight factors for individual measurement samples.
%                           1xN matrix. The default are equal weights=1.
%                           If provided, biases the estimation towards samples
%                           with higher weight but also increases estimator 
%                           variance.
%
%   OUTPUT: X_filt             ->  mxN matrix with estimated states by
%                                 Kalman filter
%           X_smooth         ->  mxN matrix with estimated states by 
%                                 Rauch-Tung-Striebel smoother.
%           MeasNoiseFilt     ->  Filtered measurement error estimates
%           MeasNoiseSmoothed ->  Smoothed measurement error estimates
%           innovation        ->  Time course of the innovation signal / filter prediction errors
%           Energy            ->  energy = -log p(y)
%                                 This is a measure of model plausibility and can be used to compare different models
%                                 or data sets.Samples with NaN-values or R=inf are ignored for this calculation.
%
%
%   Eike Petersen, 2015-2021

%% INPUT HANDLING

p = inputParser;

% Positional arguments
addRequired(p, 'Y', @ismatrix);
addRequired(p, 'A', @(A) isnumeric(A) && ndims(A) <= 3);
addRequired(p, 'C', @(C) isnumeric(C) && ndims(C) <= 3);
addRequired(p, 'Q', @(Q) isnumeric(Q) && ndims(Q) <= 3);
addRequired(p, 'R', @(R) isnumeric(R) && ndims(R) <= 3);
addOptional(p, 'x0', [], @(x0) isnumeric(x0) && ismatrix(x0) && ~any(isnan(x0)));
addOptional(p, 'P0', [], @(P0) isnumeric(P0) && ismatrix(P0) && ~any(isnan(P0(:))));
% Name-value arguments
addParameter(p, 'handle_nans', true, @islogical);
addParameter(p, 'xmin', [], @isnumeric);
addParameter(p, 'xmax', [], @isnumeric);
addParameter(p, 'sample_weights', [], @isnumeric);

parse(p, Y, A, C, Q, R, varargin{:})
handle_nans = p.Results.handle_nans;

if isempty(p.Results.x0)
    x0 = zeros(size(A, 1), 1);
else
    x0 = p.Results.x0;
end

if isempty(p.Results.P0)
    P0 = 1e10 * eye(size(A, 1));
else
    P0 = p.Results.P0;
end

if isempty(p.Results.xmin)
    xmin = -inf * ones(size(x0));
else
    xmin = make_col_vec(p.Results.xmin);
end

if isempty(p.Results.xmax)
    xmax = inf * ones(size(x0));
else
    xmax = make_col_vec(p.Results.xmax);
end

m = length(x0);
[r, N] = size(Y);

if isempty(p.Results.sample_weights)
    sample_weights = ones(N, 1);
    non_standard_weights = false;
else
    sample_weights = p.Results.sample_weights;
    if all(sample_weights == 1)
        non_standard_weights = false;
    else
        non_standard_weights = true;
    end
end


%% INITIALIZATION

% State transition matrix
function Ai = get_Ai(ii)
    if ndims(A) == 3
        Ai = A(:, :, ii);
    else
        Ai = A;
    end
end

% Measurement matrix
function Ci = get_Ci(ii)
    if ndims(C) == 3
        Ci = C(:, :, ii);
        assert(~any(isinf(Ci(:))))
    else
        Ci = C;
        assert(~any(isinf(Ci(:))))
    end
end

% Measurement noise covariance
assert(size(R, 2) == r);
assert(size(R, 1) == 1 || size(R, 1) == N);

function Ri = get_Ri(ii)
    if size(R, 1) > 1
        Ri = diag(R(ii, :));
    else
        Ri = diag(R);
    end
    assert(any(diag(Ri) > 0), 'Zero measurement noise cannot be handled by the backward information filter.');
end

% Process noise covariance
if ndims(Q) == 1
    % single diagonal
    assert(length(Q) == m);
elseif ismatrix(Q)
    assert(size(Q, 1) == 1 || size(Q, 1) == m || size(Q, 1) == N);
    assert(size(Q, 2) == m);
else
    assert(ndims(Q) == 3)
    assert(size(Q, 3) == N);
    assert(size(Q, 1) == m && size(Q, 2) == m);
end

function Qi = get_Qi(ii)
    if ndims(Q) == 3
        Qi = Q(:, :, ii);
    else
        if all(size(Q) == [m, m])
            Qi = Q;
        elseif all(size(Q) == [1, m])
            Qi = diag(Q);
        elseif all(size(Q) == [N, m])
            Qi = diag(Q(ii, :));
        end
    end
end

P_minus_ii = P0;

x_minus = x0(:);

X_filt = zeros(m, N);
P_plus = zeros(m, m, N);
X_minus = zeros(m, N);
P_minus = zeros(m, m, N);

if non_standard_weights
    P_minus_ii_w = P0;
    P_minus_w = zeros(m, m, N);
end

if nargout > 2
    P_filt = zeros(m, m, N);
end

if nargout > 4
    MeasNoiseFilt = zeros(r, N);
end

innov_cov = zeros(r,r);
ydiff = zeros(r,1);

if nargout > 6
    innovation = zeros(r, N);
end

if nargout > 7
    % use energy function from Särkkä book, phi(params) = -log p(y | params) - log p(params)
    % initialize with phi = -log p(params) = 0, i.e. assume a uniform prior.
    energy = 0; % energy calculated during forward pass
end


%% FILTERING

for ii = 1:N

    Ai = get_Ai(ii);
    Ci = get_Ci(ii);
    Qi = get_Qi(ii);
    Ri = get_Ri(ii);

    X_minus(:,ii) = x_minus;
    P_minus(:,:,ii) = P_minus_ii';
    x_plus = x_minus;
    P_plus_ii = P_minus_ii;

    if non_standard_weights
        P_plus_ii_w = P_minus_ii_w;
        P_minus_w(:,:,ii) = P_minus_ii_w';
    end
    
    % -----------
    % UPDATE STEP
    % -----------
    
    if nargout > 6
        innovation(:, ii) = Y(:, ii) - Ci * x_plus;
    end
    
    % Sequential updates
    for j = 1:r
        
        innov_cov(j,j) = Ci(j,:) * P_plus_ii * Ci(j,:)' + Ri(j,j) / sample_weights(ii);

        K = P_plus_ii * Ci(j,:)' / innov_cov(j,j);
        ydiff(j) = Y(j,ii) - Ci(j,:) * x_plus;
        
        % Output NaN handling
        if handle_nans && (any(isnan(K)) || isnan(ydiff(j)))
            % Something is unknown, hence assume Ri -> inf => K -> 0.
            K = zeros(size(K));
            ydiff(j) = 0; % value doesn't matter
            KC = zeros(size(K*Ci(j, :))); % assuming Ci(j, :) < inf; see assertion above
        else
            KC = K * Ci(j, :);
        end
        
        x_plus = x_plus + K * ydiff(j);
        
        % Joseph stabilized Kalman covariance matrix
        if ~isinf(Ri(j, j) / sample_weights(ii))
            P_plus_ii = (eye(m) - KC) * P_plus_ii * (eye(m) - KC)' + K * Ri(j,j) * K' / sample_weights(ii);
        else
            % The last summand K*Ri(j,j)*K' is what causes problems: this
            % is 0*inf*0=nan.
            % However, the analytical limit of this term is 0, so we just
            % omit it.
            P_plus_ii = (eye(m) - KC) * P_plus_ii * (eye(m) - KC)';
        end

        % Enforce symmetry of covariance matrix
        P_plus_ii = (P_plus_ii + P_plus_ii') / 2;
        
        if non_standard_weights
            % Joseph stabilized Kalman covariance matrix
            if ~isinf(Ri(j, j))
                P_plus_ii_w = (eye(m) - KC) * P_plus_ii_w * (eye(m) - KC)' + K * Ri(j,j) * K';
            else
                % The last summand K*Ri(j,j)*K' is what causes problems: this
                % is 0*inf*0=nan.
                % However, the analytical limit of this term is 0, so we just
                % omit it.
                P_plus_ii_w = (eye(m) - KC) * P_plus_ii_w * (eye(m) - KC)';
            end
            P_plus_ii_w = (P_plus_ii_w + P_plus_ii_w') / 2;
        end
    end

    % State projection to ensure keeping constraints satisfied.
    % This implements the approach detailed in section 2.4 of
    % "Kalman filtering with state constraints: a survey of linear and
    % nonlinear algorithms" by Dan Simon, IET Control Theory and
    % Applications, 2009.
    x_plus = min([xmax, max([xmin, x_plus], [], 2)], [], 2);
    
    X_filt(:,ii) = x_plus;
    P_plus(:,:,ii) = P_plus_ii';
    
    if non_standard_weights
        P_filt(:, :, ii) = P_plus_ii_w';
    else
        P_filt(:, :, ii) = P_plus_ii';
    end
    
    if nargout > 4
        MeasNoiseFilt(:, ii) = Y(:, ii) - Ci * x_plus;
    end

    if nargout > 7 && ~(any(isnan(ydiff)) || any(isnan(Ci(:))))
        % update energy function: phi_k(params) = phi_{k-1}(params) - log p(y(k) | y(1:k-1), params)
        % [Eqs. 12.11+12.38 in Särkkä book]
        if non_standard_weights
            cov_y_minus = Ci * P_minus_ii_w * Ci' + Ri;
        else
            cov_y_minus = Ci * P_minus_ii * Ci' + Ri;
        end
        energy = energy + 0.5 * (log(det(2*pi*cov_y_minus)) + ydiff' * (cov_y_minus \ ydiff));
    end
    
    % ---------------
    % PREDICTION STEP
    % ---------------
    
    x_minus = Ai * x_plus;
    P_minus_ii = Ai * P_plus_ii * Ai' + Qi;
   
    if non_standard_weights
        P_minus_ii_w = Ai * P_plus_ii_w * Ai' + Qi;
    end
end


if nargout > 1
    %% SMOOTHING
    % ----------
    % Two-Filter Smoother, following D. Simon, Optimal State Estimation, p. 283
    % The backward filter is also a linear Kalman filter, but implemented in the form of an
    % information filter that passes the information matrix and the information matrix-weighted mean
    % from time step to time step.
    
    % s=I*x is the weighted (backwards-filtered) mean, I = P^-1, the information matrix
    % Initialize uninformative prior. The forward and backward runs are completely independent of
    % one another!    
    s_minus = zeros(m, 1);
    % True zero initialization leads to numerical problems in the following.
    % (One could find an implementation able to handle this though, if it really matters.)    
    I_minus = 1e-10 * eye(size(A, 1));
    
    if non_standard_weights
        I_minus_w = I_minus;
    end
    
    X_smooth = zeros(m, N);

    % initialize last time step with filter results
    X_smooth(:, N) = X_filt(:, N);

    if nargout > 3
        P_smooth = zeros(m, m, N);
        P_smooth(:, :, N) = P_filt(:, :, N); 
    end
    
    if nargout > 5
        MeasNoiseSmoothed = zeros(size(Y, 1), N);
        MeasNoiseSmoothed(:, N) = MeasNoiseFilt(:, N);
    end    
    
    % Set up warning handling to prevent spamming the command line in case of ill conditioning
    warning('off', 'MATLAB:illConditionedMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');
    singular_matrix_warnings_found = 0;
    lastwarn('');
    for ii = N:-1:2
        Ci = get_Ci(ii);
        Ri = get_Ri(ii);
        
        % update
        Ri_nan = Ri;
        Yi_nan = Y(:, ii);
        Ci_nan = Ci;
        if any(isnan(Yi_nan)) || any(isnan(Ci_nan(:)))
            % Note that in the multivariate case, depending on the model, it might still be possible
            % to obtain information from a measurement with NaN entries.
            for jj = 1:length(Yi_nan)
                if isnan(Yi_nan(jj))
                    Ri_nan(jj, jj) = inf;
                    Yi_nan(jj) = 0;
                end
            end
            [rows_with_nan, ~] = find(isnan(Ci_nan));
            Ri_nan(sub2ind(size(Ri_nan), unique(rows_with_nan), unique(rows_with_nan))) = inf;
            Ci_nan(isnan(Ci_nan)) = 0;
        end
        s_filt = s_minus + sample_weights(ii) * Ci_nan' / Ri_nan * Yi_nan;
        CRC =  Ci_nan' / Ri_nan * Ci_nan;
        I_filt = I_minus + sample_weights(ii) * CRC;
        assert(~any(isnan(I_filt(:))));
        
        if non_standard_weights
            if sample_weights(ii) == 0
                I_filt_w = I_minus_w;
            else
                III = I_minus / I_minus_w * I_minus';
                IwCRC = I_minus + sample_weights(ii) * CRC;
                I_filt_w = inv(IwCRC \ III / IwCRC' + sample_weights(ii)^2 * (IwCRC \ CRC / IwCRC'));
                assert(~any(isnan(I_filt_w(:))));
            end
        end
        
        % predict (backwards)
        Qi = get_Qi(ii-1);
        
        if ~any(Qi(:))
            % Qi == 0. In this case, we can simplify the calculation of I_minus significantly
            Ai = get_Ai(ii-1);
            I_minus = Ai' * I_filt * Ai;
            
            if non_standard_weights
                I_minus_w = Ai' * I_filt_w * Ai;
            end
            
            s_minus = I_minus / Ai / I_filt * s_filt;
        else
            % use standard text book formula
            Ai_inv = inv(get_Ai(ii-1));
            I_minus = inv(Ai_inv / I_filt * Ai_inv' + Ai_inv * Qi * Ai_inv');
            if any(isnan(I_minus(:))) || any(isinf(I_minus(:)))
                % Probably, I_filt got non-SPD due to numerical inaccuracies. Attempt to recover...
                I_filt = nearestSPD(I_filt);
                % This might already fix things...
                I_minus = inv(Ai_inv / I_filt * Ai_inv' + Ai_inv * Qi * Ai_inv');
                while any(isnan(I_minus(:))) || any(isinf(I_minus(:)))
                    % Nope, still not working. Iteratively add multiples of the eigen space associated with the 
                    % smallest eigenvector until the result is non-nan/non-inf.
                    [V, D] = eig(I_filt);
                    [mineig, idx] = min(diag(D));
                    maxeig = max(diag(D));
                    I_filt = nearestSPD(I_filt + max([10*mineig, 1e-15*maxeig]) * V(:, idx) * V(:, idx)');
                    I_minus = inv(Ai_inv / I_filt * Ai_inv' + Ai_inv * Qi * Ai_inv');
                end
                assert(~any(isnan(I_minus(:))) && ~any(isinf(I_minus(:))));
            end

            if non_standard_weights
                I_minus_w = inv(Ai_inv / I_filt_w * Ai_inv' + Ai_inv * Qi * Ai_inv');
                if any(isnan(I_minus_w(:))) || any(isinf(I_minus_w(:)))
                    % Probably, I_filt_w got non-SPD due to numerical inaccuracies. Attempt to recover...
                    I_filt_w = nearestSPD(I_filt_w);
                    % Now everything should be fine
                    I_minus_w = inv(nearestSPD(Ai_inv / I_filt_w * Ai_inv' + Ai_inv * Qi * Ai_inv'));
                    while any(isnan(I_minus_w(:))) || any(isinf(I_minus_w(:)))
                        % Nope, still not working. Iteratively add multiples of the eigen space associated with the 
                        % smallest eigenvector until the result is non-nan/non-inf.
                        [V, D] = eig(I_filt_w);
                        [mineig, idx] = min(diag(D));
                        maxeig = max(diag(D));
                        I_filt_w = nearestSPD(I_filt_w + max([10*mineig, 1e-15*maxeig]) * V(:, idx) * V(:, idx)');
                        I_minus_w = inv(nearestSPD(Ai_inv / I_filt_w * Ai_inv' + Ai_inv * Qi * Ai_inv'));
                    end                  
                    assert(~any(isnan(I_minus_w(:))) && ~any(isinf(I_minus_w(:))));
                end
            end
            
            s_minus = I_minus * Ai_inv / I_filt * s_filt;
        end      
        
        if ~all(isinf([xmin; xmax]))
            % Project into feasible region
            x_minus = I_minus \ s_minus;
            if any(x_minus < xmin) || any(x_minus > xmax)
                x_minus = min([xmax, max([xmin, x_minus], [], 2)], [], 2);
                s_minus = I_minus * x_minus;
            end
        end    
        
        % Calculate smoothed quantities
        I_plus = inv(P_plus(:, :, ii-1));
        I_smooth_internal = I_plus + I_minus;
        X_smooth(:, ii-1) = I_smooth_internal \ (P_plus(:, :, ii-1) \ X_filt(:, ii-1) + s_minus);

        if non_standard_weights
            P_smooth(:, :, ii-1) = ...
                I_smooth_internal \ (I_plus * P_filt(:, :, ii-1) * I_plus' + ...
                I_minus / I_minus_w * I_minus') / I_smooth_internal;
        else
            P_smooth(:, :, ii-1) = inv(I_smooth_internal);
            % Test whether the more complex covariance formula above works correctly for the unweighted case.
            P_smooth_w_test = I_smooth_internal \ (I_plus * P_filt(:, :, ii-1) * I_plus' + ...
                    I_minus / I_minus * I_minus') / I_smooth_internal;
            I_smooth_cond = rcond(I_smooth_internal);
            P_plus_cond = rcond(P_plus(:, :, ii-1));
            I_minus_cond = rcond(I_minus);
            if min([I_smooth_cond, P_plus_cond, I_minus_cond]) > 1e-8
                % heuristic bound
                max_abs_err = nanmax(abs(P_smooth(:, :, ii-1) - P_smooth_w_test), [], 'all');
                max_rel_err = nanmax((P_smooth(:, :, ii-1) - P_smooth_w_test) ./ P_smooth(:, :, ii-1), [], 'all');
                assert(max_abs_err < 1e-14 || max_rel_err < 1e-3);
            end
        end
        
        % State projection to ensure keeping constraints satisfied.
        % This implements the approach detailed in section 2.4 of
        % "Kalman filtering with state constraints: a survey of linear and
        % nonlinear algorithms" by Dan Simon, IET Control Theory and
        % Applications, 2009.
        X_smooth(:, ii-1) = min([xmax, max([xmin, X_smooth(:, ii-1)], [], 2)], [], 2);   
        
        if nargout > 5
            Ci = get_Ci(ii);
            MeasNoiseSmoothed(:, ii-1) = Y(:, ii-1) - Ci * X_smooth(:, ii-1);
        end
        
        [~, msgidlast] = lastwarn;
        if strcmp(msgidlast,'MATLAB:illConditionedMatrix') || strcmp(msgidlast,'MATLAB:nearlySingularMatrix') ...
                || strcmp(msgidlast, 'MATLAB:singularMatrix')
            singular_matrix_warnings_found = singular_matrix_warnings_found + 1;
        else
            disp(msgidlast)
        end

        lastwarn('');
    end  % end smoother loop
  
    if singular_matrix_warnings_found > 0
        warning('Matrix close to singular or singular during %d smoothing steps.', singular_matrix_warnings_found);
    end
    warning('on', 'MATLAB:illConditionedMatrix');
    warning('on', 'MATLAB:nearlySingularMatrix');
    warning('on', 'MATLAB:singularMatrix');

end

end