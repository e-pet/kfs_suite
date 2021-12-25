function grid_array = ndgridarr(n, varargin)

    assert(length(varargin) == 1 || length(varargin) == n);

    grid_cells = cell(1, n);
    [grid_cells{:}] = ndgrid(varargin{:});
    
    % https://stackoverflow.com/a/60714674/2207840
    grid_array = reshape(cat(n+1,grid_cells{:}),[],n);
end