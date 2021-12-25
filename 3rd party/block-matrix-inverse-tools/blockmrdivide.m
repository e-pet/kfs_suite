function R = blockmrdivide(M, subsizes, b, varargin)
% BLOCKMRDIVIDE implement matrix right-divide (see @mrdivide), as in the solution to x*M = b
% implemented as x = b / M. Here, x = blockmldivide(M, subsizes, b) where 'subsizes' is defined as
% in @blockinv. See @blockinv and @blockmldivide for further details.

R = b * blockinv(M, subsizes, varargin{:});

end