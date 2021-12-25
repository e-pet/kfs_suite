function [fig, subplot_handles] = plot_signals(signals, varargin)
%% PLOT_SIGNALS Plot signals contained in a matrix into subplots of a single figure
%
% Generates a single figure with subplots of signals contained in
% 'signals'. The length of the smaller dimension of signals is
% automatically interpreted as the number of signals to plot. All signals
% are plotted against a common x axis, and various visual tweaks are
% employed to make the plot more useful and visually appealing than the
% default plots.
%
% To compare two sets of signals, see the function compare_signals.
%
% INPUT 
% ------ POSITIONAL PARAMETERS ------
% signals:       A (2-D) matrix that contains signals of identical length 
%                to plot. It does not matter whether the signals are in 
%                rows or columns; the longer axis is always interpreted as 
%                the x axis dimension.
%                Alternatively: a cell array containing individual signals.
%                This allows using signals with different sampling rates.
%                This is the _only_ required parameter!
% time:          x vector to plot against, must have same length as
%                signals. If omitted or [], the index is used instead.
%                Alternatively: a cell array with as many entries as there
%                are signals in 'signals', where each entry specifies the
%                x axis of the corresponding signal. This second option
%                *must* be used if 'signals' is a cell array! Optional.
% signal_labels: Either (1) a single character vector, in which case this
%                vector with appended '_i' is used as the y label for each
%                signal, where i is the signal's index, or (2) a cell array
%                with the same number of entries as there are signals in
%                signals, where each entry contains the label for
%                that signal, or (3) '', in which case the signal index i is
%                used. Optional.
% xlab:          x Axis label. Optional.
% plot_title:    Overall plot title to put over the whole figure. Optional.
%
% ------ OPTIONAL PARAMETERS, PASS AS NAME/VALUE ------
% num_plot_cols: Number of subplot columns. Default is 1.
% plot_by_rows:  Boolean that specifies whether to fill subplots by rows
%                (true, default) or by columns (false).
% markers:       Either a single vector of the same length as the individual
%                signals, or a cell array with one entry for each signal.
%                Each entry can either be a vector of the same length as
%                the signals, or {}. For each non-zero entry in a vector, a
%                marker is shown in the plot. If only one vector is
%                provided, the markers are plotted in all signals;
%                otherwise they are only plotted for a single signal.
% vlines:        Same as markers, but vertical lines are plotted instead of
%                markers. Also the format is different: here, the vectors
%                should contain the x values at which to plot the lines.
%                Caution: this causes the plot to become very slow if many 
%                vlines are shown!
% vline_labels:  Labels for vlines. If vlines is a single vector, this must
%                be a cell array with one string per entry in vlines. If
%                vlines is a cell array, this must be a cell array with one
%                entry per signal as well.
% ref_sigs:      Cell array with one entry per signal. If entry (i) is
%                non-empty, it must contain a signal vector or matrix
%                with further signals that should be added to the plot of
%                signal (i).
% ref_sig_labels:Cell array with one entry per signal. If entry (i) is
%                non-empty, it must contain a cell array containing the
%                titles of all signals (including the original,
%                non-reference one, markers, vlines, and ref_sigs) plotted 
%                in subplot i.
% fig_title:     Figure title. If not provided, plot_title is used.
% logy:          Should individual signals be shown with a logarithmic y axis? If yes, provide an
%                array with the indices of the corresponding signals, i.e. [2,3, 7]. Default is []
%                (none).
% link_axes:     Axes across which to link subplots: 'x', 'y', 'xy', ''. 'x' is default. Linking the
%                y axis is not possible when logy is non-empty.
% linespec:      Standard plot linespec. Default is '-'.
%
% OUTPUTS        fig: The generated figure.
%    subplot_handles: Vector containing handles for the generated subplots.
%
% EXAMPLES
%   t = 1:1000;
%   mat = [1:1000; sin(2*pi*t/100)];
%   % This is what I originally developed this for: easy and direct plotting of signals in a matrix
%   plot_signals(mat);  
%
%   % Here's a more complex example of what's also possible with this function.
%   ref_sigs = {};
%   ref_sigs{2} = repmat(linspace(-1, 1, 100), 1, 10);
%   ref_sig_labels = {};
%   ref_sig_labels{2}={'sine', 'phase'};
%   vlines = 100:100:900;
%   plot_signals(mat, t, {'Signal A', 'Signal B'}, 'Time (s)', 'Demo Title', ...
%                'vlines', vlines, ...
%                'ref_sigs', ref_sigs, ...
%                'ref_sig_labels', ref_sig_labels);
%
% Eike Petersen, 2019-2021
% University of Lübeck, Institute for Electrical Engineering in Medicine (2019-2021)
% Danmarks Tekniske Universitet / Technical University of Denmark, DTU Compute (2021-)
%


%% Input handling
p = inputParser;

% Positional arguments
addRequired(p, 'signals', @ismatrix);
addOptional(p, 'time', [], @(time) isempty(time) || isvector(time));
addOptional(p, 'signal_labels', [], @(par) isempty(par) || iscell(par) || ischar(par));
addOptional(p, 'xlab', [], @(par) isempty(par) || ischar(par));
addOptional(p, 'plot_title', [], @(par) isempty(par) || ischar(par));
% Name-value arguments
addParameter(p, 'num_plot_cols', 1, @isnumeric);
addParameter(p, 'plot_by_rows', true, @islogical);
check_vec_or_cell_of_vecs = @(par) isempty(par) || isvector(par) || (iscell(par) && isvector(par{1}));
addParameter(p, 'markers', {}, check_vec_or_cell_of_vecs);
addParameter(p, 'vlines', {}, check_vec_or_cell_of_vecs);
addParameter(p, 'vline_labels', {}, @iscell);
addParameter(p, 'ref_sigs', {}, @iscell);
addParameter(p, 'ref_sig_labels', {}, @iscell);
addParameter(p, 'fig_title', '', @ischar);
addParameter(p, 'logy', [], @(logy) isempty(logy) || isvector(logy));
addParameter(p, 'link_axes', 'x', @ischar);
addParameter(p, 'linespec', '-', @ischar);

parse(p, signals, varargin{:})
struct2vars(p.Results);

% Make sure data are in rows
if ~iscell(signals)
    [m, n] = size(signals);
    if m > n
        signals = signals';
    end
end

if isempty(time)
    time = 1:size(signals, 2);
end

if isempty(signal_labels)
    signal_labels = '';
end

if isempty(xlab)
    xlab = '';
end

if isempty(plot_title)
    plot_title = '';
end

if isempty(fig_title)
    fig_title = plot_title;
end

if ~isempty(logy)
    assert(~strcmp(link_axes, 'y') && ~strcmp(link_axes, 'xy'));
end

num_plot_rows = ceil(size(signals, 1) / num_plot_cols);
nsig = size(signals, 1);

%% Set up figure
if strcmp(fig_title, '')
    fig = figure;
else
    fig = figure('name', fig_title, 'NumberTitle','off');
end

% Changing some defaults to make plots look nicer
set(fig,   'DefaultAxesXGrid', 'on', ...
           'DefaultAxesYGrid', 'on', ...
           'DefaultAxesFontWeight', 'normal', ...
           'DefaultAxesFontSize', 12, ...
           'DefaultAxesTitleFontWeight', 'normal', ...
           'DefaultAxesTitleFontSizeMultiplier', 1, ...
           'DefaultTextFontName', 'Cambria', ...
           'DefaultLineLineWidth', 1.5) ;

if plot_by_rows == false
    index = reshape(1:nsig, num_plot_cols, num_plot_rows).';
else 
    index = 1:nsig;
end

tl = tiledlayout(fig, num_plot_rows, num_plot_cols, 'TileSpacing', 'none', 'Padding', 'none');
subplot_handles = zeros(num_plot_rows * num_plot_cols, 1);
for ii = 1:(num_plot_rows*num_plot_cols)
    subplot_handles(ii) = nexttile;
end

if iscell(time)
    xlims = [min(time{1}), max(time{1})];
else
    xlims = [min(time), max(time)];
end

%% Plot individual signals
for ii = 1:nsig
    
    ax = subplot_handles(index(ii));
    if iscell(signals)
        signal = signals{ii};
        t = time{ii};
    else
        signal = signals(ii, :);
        t = time;
    end
    
    if ismember(ii, logy)
        semilogy(ax, t, signal, linespec);
    else
        plot(ax, t, signal, linespec);
    end
    
    if ~isempty(ref_sigs)
        if ~isempty(ref_sigs{ii})
            hold(ax, 'on');
            % Make sure data are in rows
            [m, n] = size(ref_sigs{ii});
            if m > n
                ref_sigs{ii} = ref_sigs{ii}';
                [m, ~] = size(ref_sigs{ii});
            end
            for jj = 1:m
                if ismember(ii, logy)
                    semilogy(ax, t, ref_sigs{ii}(jj, :), linespec);
                else
                    plot(ax, t, ref_sigs{ii}(jj, :), linespec);
                end
            end
            

        end
    end
    
    if ~isempty(ref_sig_labels) && ~isempty(ref_sig_labels{ii})
        leg = legend(ax, ref_sig_labels{ii}{:}, 'AutoUpdate', 'off');
        % Enable clicking on axis to hide lines
        leg.ItemHitFcn = @hileline;
    end
    
    if ~isempty(markers)
        hold(ax, 'on');
        if ~iscell(markers)
            scatter(ax, t(markers~=0), signal(markers~=0), 'ro');
        else
            scatter(ax, t(markers{ii}~=0), signal(markers{ii}~=0), 'ro');
        end
    end
    
    if ~isempty(vlines)
        if ~iscell(vlines)
            add_vlines(subplot_handles, vlines, 'k-', vline_labels);
        else
            add_vlines(ax, vlines{ii}, 'k-', vline_labels{ii});
        end
    end
    
    % Only show x axis label and x tick labels in bottom plots
    if index(ii) <= nsig - num_plot_cols
        set(ax, 'XTickLabelMode', 'manual') 
        set(ax, 'XTickLabel', {})        
    else
        xlabel(ax, xlab);
    end
    
    % Adjust axis limits
    ylim_auto = ylim(ax);
    ymin_data = min(signal);
    ymax_data = max(signal);
    yrange = ymax_data - ymin_data;
    ymin_plot = min(ylim_auto(1), ymin_data - 0.1*yrange);
    ymax_plot = max(ylim_auto(2), ymax_data + 0.1*yrange);
    ylim(ax, [ymin_plot, ymax_plot]);
    xlim(ax, xlims);
    
    % Add y label
    if iscell(signal_labels)
        sig_name = signal_labels{ii};
    else
        sig_name = [signal_labels, '_', num2str(ii)];
    end
    ylabel(ax, sig_name);
    
end

linkaxes(subplot_handles(1:nsig), link_axes);

if nargin >= 5
    title(tl, plot_title);
end

end


%% Helper functions

function add_vlines(subplot_handles, x_positions, linespec, vline_labels)

    version_str = version('-release');
    version_year = sscanf(version_str(1:end-1), '%d');
    if version_year >= 2019 || strcmp(version_str, 'R2018b')
        % can use builtin functions which are available as of R2018b
        if version_year >= 2021
            % there is now a vectorized version, yay
            for kk = 1:length(subplot_handles)
                xline(subplot_handles(kk), x_positions, linespec, vline_labels);
            end
        else
            for idx = 1:length(x_positions)
                if ~isempty(vline_labels)
                    label = vline_labels(idx);
                else
                    label = '';
                end
                for kk = 1:length(subplot_handles)
                    xline(subplot_handles(kk), x_positions(idx), linespec, label);
                end
            end
        end
    else
        % use file exchange function
        vline2(subplot_handles, x_positions, linespec, vline_labels);
    end
end


function hileline(src, event)
    % This callback toggles the visibility of the line

    if strcmp(event.Peer.Visible,'on')   % If current line is visible
        event.Peer.Visible = 'off';      %   Set the visibility to 'off'
    else                                 % Else
        event.Peer.Visible = 'on';       %   Set the visibility to 'on'
    end
end
