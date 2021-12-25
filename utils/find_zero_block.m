function upper_block_size = find_zero_block(A)
% Look for square zero block either top left or bottom right in square matrix A.
%
% This enables a representation of A as a 2x2 block matrix with a zero block.
% If both top left and bottom right are zero blocks, return the larger of
% the two. If both are equal, choose the lower right block.
%
% In particular, the size of the top left block is returned that yields the
% selected block partition. 
% If upper_block_size == size(A, 1), no zero block was found.
% If upper_block_size == 0, the whole matrix is zero.

assert(ismatrix(A))
assert(size(A, 1) == size(A, 2))

upper_block_size = find_top_left_square_zero_block(A);
lower_block_size = find_top_left_square_zero_block(flip(flip(A, 2)));

if lower_block_size >= upper_block_size
    upper_block_size = size(A, 1) - lower_block_size;
end

end


function upper_zero_block_size = find_top_left_square_zero_block(A)
% This is probably not the most efficient way of doing this...
    test_index = 1;
    while ~(any(A(test_index, 1:test_index)) || any(A(1:test_index, test_index)))
        test_index = test_index + 1;
    end
    upper_zero_block_size = test_index - 1;
end