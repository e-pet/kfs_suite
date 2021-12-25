function [Xfilt, Xsmoothed, MeasNoiseFilt, MeasNoiseSmoothed, iter, innovation] = KF_RTS_NL(Y, A, C, Q, R, varargin)
% KF_RTS_NL sequential Kalman Filter and Smoother for nonlinear systems
% Linear and nonlinear Kalman Filter and Smoother; sequential implementation of Joseph
% stabilized, symmetrized form. Only works for diagonal measurement noise
% covariance matrices R!
%
% To use a time-varying linear system description, pass 3d arrays for A, C, and/or Q,
% where the last index iterates over sample indices and/or an array for R,
% where the rows indicate different samples.
%
% To use a nonlinear system, pass a function handle for A and/or C. An iterative
% sigma point filter/smoother is used in that case. The quadrature scheme / sigma point
% selection scheme can be specified; the default is the standard unscented transform.
% By default, multiple iterative filter/smoother runs are performed, often leading to 
% highly significant performance gains.
% To consider a time-varying nonlinear system, pass a function handle that takes two
% inputs, x and ii (the measurement index).
%
% Inputs are currently not considered.
%
% For details on the KF and its sequential implementation and the RTS see Dan Simon, Optimal State Estimation (2006), p. 150.
% For general discrete-time nonlinear filtering using sigma point methods, see Arasaratnam and
% Haykin (2007), "Discrete-Time Nonlinear Filtering Algorithms Using Gauss-Hermite Quadrature".
% The formulation of state constraints is due to section 2.4 of "Kalman filtering with state constraints: a survey of 
% linear and nonlinear algorithms" by Dan Simon, IET Control Theory and Applications, 2009.
% For the generalized iterative quadrature filtering implemented here, see Herzog, Petersen, Rostalski (2019), 
% "Iterative Approximate Nonlinear Inference via Gaussian Message Passing on Factor Graphs".
% Some of the implementation details are my own. (Eike Petersen)
%
%   INPUT:  Y           ->  rxN matrix where r is the number of measurement channels and
%                           N is the length of each measured signal.
%           A           ->  System matrix.
%           C           ->  Observation matrix or function.
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
%           x0          ->  Vector of length m with initial state. (Optional)
%           P0          ->  mxm matrix with initial state covariance. (Optional)
%           handle_nans ->  handle 'nan' values in measurements or
%                           observation matrices by simply treating this
%                           measurement as completely unknown (infinite 
%                           measurement variance). Default is true.
%           xmin        ->  Lower bound for state values. (Optional)
%           xmax        ->  Upper bound for state values. (Optional)
%           quad_rule   ->  Quadrature rule to be used for linearization.
%                           (Optional)
%           return_iter ->  Instead of normal outputs, return a cell array
%                           for each output signal, where each cell
%                           contains the result of one filter/smoother
%                           iteration. (Optional)
%           max_iter    ->  Maximum number of iterations for nonlinear
%                           filter/smoother. (Default is 20.)
%      max_abs_diff_tol ->  Tolerance for maximum absolute state
%                           difference between iterations. If the max abs
%                           diff is below this threshold, iterations are
%                           stopped. (Default is 1e-5.)

%
%   OUTPUT: Xfilt             ->  mxN matrix with estimated states by
%                                 Kalman filter
%           Xsmoothed         ->  mxN matrix with estimated states by 
%                                 Rauch-Tung-Striebel smoother.
%           MeasNoiseFilt     ->  Filtered measurement error estimates
%           MeasNoiseSmoothed ->  Smoothed measurement error estimates
%           iter              ->  Number of iterations the nonlinear
%                                 smoother has been run for.
%           innovation        ->  Time course of the innovation signal / filter prediction errors
%
%  Note that in the nonlinear, iterative case, only filter/smoother results
%  from the last iteration are returned! This means that the return
%  'filter' results are _not_ "real-time". If you want to see results of
%  all iterations, pass 'return_iter' = true.
%
%  This is also true for the innovation signal - it's also from the last iteration.
%
%
%   Eike Petersen, 2015-2021

%% INPUT HANDLING

p = inputParser;
is_mat_or_list_of_mats = @(M)isnumeric(M) && ndims(M) <= 3;
is_mat_or_list_of_mats_or_func_handle = ...
    @(M) is_mat_or_list_of_mats(M) || isa(M, 'function_handle');

% Positional arguments
addRequired(p, 'Y', @ismatrix);
addRequired(p, 'A', is_mat_or_list_of_mats_or_func_handle);
addRequired(p, 'C', is_mat_or_list_of_mats_or_func_handle);
addRequired(p, 'Q', is_mat_or_list_of_mats);
addRequired(p, 'R', is_mat_or_list_of_mats);
% Name-value arguments
addOptional(p, 'x0', [], @(x) isnumeric(x) && isvector(x));
addOptional(p, 'P0', [], @(x) isnumeric(x) && ismatrix(x));
addParameter(p, 'handle_nans', true, @islogical);
addParameter(p, 'xmin', [], @(x) isnumeric(x) && isvector(x));
addParameter(p, 'xmax', [], @(x) isnumeric(x) && isvector(x));
addParameter(p, 'quad_rule', [], @(f) isa(f, 'function_handle'));
addParameter(p, 'return_iter', false, @islogical);
addParameter(p, 'max_iter', 20, @(x) isscalar(x) && round(x) == x && x >= 1);
addParameter(p, 'max_abs_diff_tol', 1e-5, @(x) isscalar(x) && x > 0);

parse(p, Y, A, C, Q, R, varargin{:})

handle_nans = p.Results.handle_nans;
return_iter = p.Results.return_iter;
max_iter = p.Results.max_iter;
max_abs_diff_tol = p.Results.max_abs_diff_tol;

if isempty(p.Results.x0)
    if ~isa(A, 'function_handle')
        x0 = zeros(size(A, 1), 1);
    else
        error('Must provide value for x0 if A is a function handle.');
    end
else
    x0 = p.Results.x0;
end

m = length(x0);

if isempty(p.Results.P0)
    P0 = 1e7 * eye(m);
else
    P0 = p.Results.P0;
end

if isempty(p.Results.xmin)
    xmin = -inf * ones(size(x0));
else
    xmin = p.Results.xmin;
end

if isempty(p.Results.xmax)
    xmax = inf * ones(size(x0));
else
    xmax = p.Results.xmax;
end

if isempty(p.Results.quad_rule)
    quad_rule = @sigma_points_classic;
else
    quad_rule = p.Results.quad_rule;
end


%% INITIALIZATION

[r, N] = size(Y);

% Process noise covariance
if ndims(Q) == 1
    % single diagonal
    assert(length(Q) == m);
elseif ndims(Q) == 2
    assert(size(Q, 1) == 1 || size(Q, 1) == m || size(Q, 1) == N);
    assert(size(Q, 2) == m);
else
    assert(ndims(Q) == 3)
    assert(size(Q, 3) == N);
    assert(size(Q, 1) == m && size(Q, 2) == m);
end

Q_upper_block_size = [];

if nargout > 5
    innovation = zeros(r, N);
end

function Qi = get_Qi(ii)
    if ndims(Q) == 3
        Qi = Q(:, :, ii);
        % Automatically inferring block size anew each iteration is very
        % inefficient. Set block size to full.
        Q_upper_block_size = size(Qi, 1);
    else
        if all(size(Q) == [m, m])
            Qi = Q;
            if isempty(Q_upper_block_size)
                Q_upper_block_size = find_zero_block(Qi);
            end
        elseif all(size(Q) == [1, m])
            Qi = diag(Q);
            if isempty(Q_upper_block_size)
                Q_upper_block_size = find_zero_block(Qi);
            end            
        elseif all(size(Q) == [N, m])
            Qi = diag(Q(ii, :));
            % Automatically inferring block size anew each iteration is very
            % inefficient. Set block size to full.
            Q_upper_block_size = size(Qi, 1);            
        end
    end
end


function [Ai, bi, Qi] = get_state_transition_model(ii, x_marginal, P_marginal, x_marginal_old, P_marginal_old)
    if isa(A, 'function_handle')
        if nargin(A) == 2
            Afunci = @(x) A(x, ii);
        elseif nargin(A) == 1
            Afunci = A;
        else
            error('A function handle takes unexpected number of inputs.');
        end
        
        if nargin > 3
            damping = 0.8;
            %damping = 1;
            x_marginal = damping * x_marginal + (1-damping) * x_marginal_old;
            P_marginal = damping * P_marginal + (1-damping) * P_marginal_old;
        end
        
        [Ai, bi, Qi] = statistical_linearization(Afunci, x_marginal, P_marginal, quad_rule);
        % Combine linearization noise with modeled process noise
        Qi = Qi + get_Qi(ii);
        % Automatically inferring block size anew each iteration is very
        % inefficient. Set block size to full.        
        Q_upper_block_size = size(Qi, 1);
    else
        % basic linear filter/smoother
        % state transition matrix
        if ndims(A) == 3
            Ai = A(:, :, ii);
        else
            Ai = A;
        end
        
        % process noise covariance
        Qi = get_Qi(ii);
        % process noise mean
        bi = zeros(size(x_marginal));
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
end


function [Ci, di, Ri] = get_measurement_model(ii, x_marginal, P_marginal, x_marginal_old, P_marginal_old)
    if isa(C, 'function_handle')
        if nargin(C) == 2
            Cfunci = @(x) C(x, ii);
        elseif nargin(C) == 1
            Cfunci = C;
        else
            error('C function handle takes unexpected number of inputs.');
        end
        
        if nargin > 3
            damping = 0.8;
            %damping = 1;
            x_marginal = damping * x_marginal + (1-damping) * x_marginal_old;
            P_marginal = damping * P_marginal + (1-damping) * P_marginal_old;
        end
        
        [Ci, di, Ri] = statistical_linearization(Cfunci, x_marginal, P_marginal, quad_rule);
        % Combine linearization noise with modeled process noise
        Ri = Ri + get_Ri(ii);
    else
        % basic linear filter/smoother
        % measurement matrix
        if ndims(C) == 3
            Ci = C(:, :, ii);
            assert(~any(isinf(Ci(:))))
        else
            Ci = C;
            assert(~any(isinf(Ci(:))))
        end
        
        % measurement noise covariance
        Ri = get_Ri(ii);
        % measurement noise mean
        di = zeros(size(x_marginal));
    end
end

Xfilt = zeros(m, N);
Phat = zeros(m, m, N);
Xbar = zeros(m, N);
Pbar = zeros(m, m, N);
MeasNoiseFilt = zeros(r, N);

Xsmoothed = nan * ones(m, N);
Psmoothed = nan * ones(m, m, N);
S = nan * ones(m, m, N);
S_is_inv = zeros(N, 1);
Cis = zeros(r, m, N);

% Initialize innovation monitoring
innov_cov = zeros(r,r);
ydiff = zeros(r,1);

if ~(isa(A, 'function_handle') || isa(C, 'function_handle'))
    % Everything is linar; no need to iterate!
    % Perform standard RTS.
    max_iter = 1;
end

max_abs_diff = inf;
max_abs_diff_last = inf;
iter = 0;
increasing_diff_steps = 0;
max_state_diff_filt = zeros(N, 1);
max_state_diff_smooth = zeros(N, 1);

% Iterate filter + smoother runs for improved linearization of nonlinear functions
% See Tronarp, Garcia-Fernandez and Särkkä (2018), Iterative Filtering and Smoothing in Nonlinear and
% Non-Gaussian Systems Using Conditional Moments
% and
% Herzog, Petersen, Rostalski (2019), 
% Iterative Approximate Nonlinear Inference via Gaussian Message Passing on Factor Graphs
%
% (The specific version with iterations both at the sample level and the overall filter/smoother
% level and additional damping is described in neither of those publications.)
%
while iter < max_iter && max_abs_diff > max_abs_diff_tol && increasing_diff_steps < 5
    %% FILTERING

    P_minus = P0;
    x_minus = x0(:);
    
    for ii = 1:N

        Xbar(:,ii) = x_minus;
        Pbar(:,:,ii) = P_minus';

        % -----------
        % UPDATE STEP
        % -----------
        
        % What are we linearizing the measurement function about?
        if iter >= 1
            x_plus = Xsmoothed(:, ii);
            P_plus = Psmoothed(:, :, ii);
        else
            x_plus = x_minus;
            P_plus = P_minus;            
        end

        if nargout > 5
            innovation(:, ii) = Y(:, ii) - Ci * x_plus;
        end        
        
        max_abs_inner_diff = inf;
        max_abs_inner_diff_last = inf;
        inner_iter = 0;
        increasing_inner_diff_steps = 0;
        if isa(C, 'function_handle') && iter == 0
            max_inner_iter = 5;
            %max_inner_iter = 1;
        else
            % Either nothing to iteratively linearize here, or we're already past the first
            % filter/smoother run
            max_inner_iter = 1;
        end
        
        while inner_iter < max_inner_iter && max_abs_inner_diff > max_abs_diff_tol && increasing_inner_diff_steps < 2
            
            % Linearize measurement function around current best guess
            if iter > 1
                [Ci, di, Ri] = get_measurement_model(ii, Xsmoothed(:, ii), Psmoothed(:, :, ii), ...
                    Xsmoothed_old(:, ii), Psmoothed_old(:, :, ii));
            elseif iter == 1
                [Ci, di, Ri] = get_measurement_model(ii, Xsmoothed(:, ii), Psmoothed(:, :, ii));
            else
                [Ci, di, Ri] = get_measurement_model(ii, x_plus, P_plus);
            end
         
            x_plus_old = x_plus;
            x_plus = x_minus;
            P_plus = P_minus;

            % Sequential updates
            for j = 1:r

                innov_cov(j,j) = Ci(j,:) * P_plus * Ci(j,:)' + Ri(j,j);
                
                K = P_plus * Ci(j,:)' / innov_cov(j,j);
                ydiff(j) = Y(j,ii) - Ci(j,:) * x_plus - di(j);

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
                if ~(isinf(Ri(j, j)) || isnan(Ri(j, j)))
                    P_plus = (eye(m) - KC) * P_plus * (eye(m) - KC)' + K * Ri(j,j) * K'; 
                else
                    % If isnan(Ri(j,j)), we assume Ri(j,j) -> inf.
                    % The last summand K*Ri(j,j)*K' is what causes problems: this is 0*inf*0=nan.
                    % However, the analytical limit of this term is 0, so we just omit it.
                    P_plus = (eye(m) - KC) * P_plus * (eye(m) - KC)';
                end            
                
                % Enforce symmetry of covariance matrix
                P_plus = (P_plus + P_plus') / 2;
            end

            % State projection to ensure keeping constraints satisfied.
            % This implements the approach detailed in section 2.4 of
            % "Kalman filtering with state constraints: a survey of linear and
            % nonlinear algorithms" by Dan Simon, IET Control Theory and Applications, 2009.
            x_plus = min([xmax, max([xmin, x_plus], [], 2)], [], 2);

            % Some iteration house-keeping...
            if inner_iter > 0
                max_abs_inner_diff = max(abs(x_plus_old - x_plus));
            end
                    
            inner_iter = inner_iter + 1;
            if max_inner_iter > 1
                fprintf('Sample %d, inner iteration %d: Max(abs(state diff))=%f.\n', ii, inner_iter, max_abs_inner_diff);

                if max_abs_inner_diff > max_abs_inner_diff_last
                    increasing_inner_diff_steps = increasing_inner_diff_steps + 1;
                else
                    increasing_inner_diff_steps = 0;
                end

                max_abs_inner_diff_last = max_abs_inner_diff;
            end
        end

        Cis(:, :, ii) = Ci;
        
        Xfilt(:, ii) = x_plus;
        Phat(:, :, ii) = P_plus';
        MeasNoiseFilt(:, ii) = Y(:, ii) - Ci * x_plus;
        
        if return_iter && iter > 0
            max_state_diff_filt(ii) = max(abs(x_plus - XfiltCell{iter}(:, ii)));
        end
        
        % ---------------
        % PREDICTION STEP
        % ---------------
        if iter > 1
            [Ai, bi, Qi] = get_state_transition_model(ii, Xsmoothed(:, ii), Psmoothed(:, :, ii), ...
                Xsmoothed_old(:, ii), Psmoothed_old(:, :, ii));
        elseif iter > 0
            [Ai, bi, Qi] = get_state_transition_model(ii, Xsmoothed(:, ii), Psmoothed(:, :, ii));
        else
            [Ai, bi, Qi] = get_state_transition_model(ii, Xfilt(:, ii), Phat(:, :, ii));
        end
        
        x_minus = Ai * x_plus + bi;
        P_minus = Ai * P_plus * Ai' + Qi;

        % Store matrix for smoothing step
        [S(:, :, ii), S_is_inv(ii)] = calc_smoothing_S(Ai, Qi, P_plus', P_minus', Q_upper_block_size);
    end

    if iter > 0
        Xsmoothed_old = Xsmoothed;
        Psmoothed_old = Psmoothed;
    end
    
    %% SMOOTHING
    % ----------
    % Rauch-Tung-Striebel-type smoothing, see, e.g., Simo Särkkä (2013):
    % "Bayesian Filtering and Smoothing"
    
    Psmoothed(:, :, N) = Phat(:, :, N);
    Xsmoothed(:, N) = Xfilt(:, N);

    MeasNoiseSmoothed = zeros(size(Y, 1), N);
    MeasNoiseSmoothed(:, N) = MeasNoiseFilt(:, N);

    % Set up warning handling to prevent spamming the command line in case of
    % ill conditioning
    warning('off', 'MATLAB:illConditionedMatrix');
    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');
    singular_matrix_warnings_found = 0;
    lastwarn('');
    
    % Keep track of differences between the current and the previous
    % smoothing run
    if iter > 0
        max_abs_diff = 0;
    else
        % This is the first run; there is nothing to compare with
        max_abs_diff = inf;
    end
    for ii = N - 1:-1:1

        xs_old = Xsmoothed(:, ii);
        if S_is_inv(ii)
            Xsmoothed(:, ii) = Xfilt(:, ii) + S(:, :, ii) \ (Xsmoothed(:, ii + 1) - Xbar(:, ii + 1));
        else
            Xsmoothed(:, ii) = Xfilt(:, ii) + S(:, :, ii) * (Xsmoothed(:, ii + 1) - Xbar(:, ii + 1));
        end
        
        % State projection to ensure keeping constraints satisfied.
        % This implements the approach detailed in section 2.4 of
        % "Kalman filtering with state constraints: a survey of linear and
        % nonlinear algorithms" by Dan Simon, IET Control Theory and
        % Applications, 2009.
        Xsmoothed(:, ii) = min([xmax, max([xmin, Xsmoothed(:, ii)], [], 2)], [], 2);   
        
        if S_is_inv(ii)
            Psmoothed(:, :, ii) = Phat(:, :, ii) - (S(:, :, ii) \ (Pbar(:, :, ii+1) - Psmoothed(:, :, ii+1))) / S(:, :, ii)';
        else
            Psmoothed(:, :, ii) = Phat(:, :, ii) - S(:, :, ii) * (Pbar(:, :, ii+1) - Psmoothed(:, :, ii+1)) * S(:, :, ii)';
        end

        if iter > 0 && ~any(isnan(xs_old))
            max_abs_diff = max(max_abs_diff, max(abs(xs_old - Xsmoothed(:, ii)), [], 'all'));
            max_state_diff_smooth(ii) = max(abs(Xsmoothed(:, ii) - xs_old));
        end
        
        MeasNoiseSmoothed(:, ii) = Y(:, ii) - Cis(:, :, ii) * Xsmoothed(:, ii);

        [~, msgidlast] = lastwarn;
        if strcmp(msgidlast,'MATLAB:illConditionedMatrix') || strcmp(msgidlast,'MATLAB:nearlySingularMatrix') ...
                || strcmp(msgidlast, 'MATLAB:singularMatrix')
            singular_matrix_warnings_found = singular_matrix_warnings_found + 1;
        else
            disp(msgidlast)
        end

        lastwarn('');
    end

    iter = iter + 1;
    if max_iter > 1
        fprintf('Iteration %d: Max(abs(state diff))=%f.\n', iter, max_abs_diff);
        
        if max_abs_diff > max_abs_diff_last
            increasing_diff_steps = increasing_diff_steps + 1;
        else
            increasing_diff_steps = 0;
        end
        
        max_abs_diff_last = max_abs_diff;
    end
    
    if return_iter
        XfiltCell{iter} = Xfilt;
        XsmoothedCell{iter} = Xsmoothed;
        MeasNoiseFiltCell{iter} = MeasNoiseFilt;
        MeasNoiseSmoothedCell{iter} = MeasNoiseSmoothed;
    end
end

if singular_matrix_warnings_found > 0
    warning('Matrix close to singular or singular during %d smoothing steps.', singular_matrix_warnings_found);
end
warning('on', 'MATLAB:illConditionedMatrix');
warning('on', 'MATLAB:nearlySingularMatrix');
warning('on', 'MATLAB:singularMatrix');

fprintf('Ran %d filter/smoother iterations.\n', iter);
if iter == max_iter
    disp('Stopping reason: max number of iterations.');
elseif max_abs_diff < max_abs_diff_tol
    disp('Stopping reason: max_abs_diff < max_abs_diff_tol.');
elseif increasing_diff_steps == 2
    disp('Stopping reason: Between-iteration diffs have increased five times in a row.');
else
	disp('Stopping reason unknown. This should not occur?');
end

if return_iter
    Xfilt = XfiltCell;
    Xsmoothed = XsmoothedCell;
    MeasNoiseFilt = MeasNoiseFiltCell;
    MeasNoiseSmoothed = MeasNoiseSmoothedCell;
end
end

function [S, S_is_inv] = calc_smoothing_S(Ai, Qi, P_plus_i, P_minus_next, Qi_upper_block_size)
    m = size(Ai, 1);
    if ~any(Qi(:))
        % Qi == 0. In this case, Pbar = P_{k+1}^- = A * P_{k-1}^T * A^T
        % and we have (analytically) that S = A^-1.
        S = Ai;
        S_is_inv = true;
    else
        if nargin < 5 || Qi_upper_block_size == m
            % use standard text book formula
            S = P_plus_i * Ai' / P_minus_next;
        else
            % Attempt to exploit (partially zero-Q) block matrix structure!
            % requires blockinv function from
            % https://github.com/wrongu/block-matrix-inverse-tools to be
            % available on the path
            S = P_plus_i * Ai' * blockinv(P_minus_next, [Qi_upper_block_size, m-Qi_upper_block_size]);
        end
        S_is_inv = false;
    end
end