Block Matrix Inversion Tools
============================

Matlab has very good built-in support for fast matrix inversion exploiting the structure of a matrix. See the [algorithms](https://www.mathworks.com/help/matlab/ref/mldivide.html#bt4jslc-6) section of the documentation on `mldivide` for more information.

The functions provided here were initially written to support a latent Gaussian Process inference implementation, where we frequently encounter large matrices which have _sub_matrices with "nice" structure, but the full matrix does not. These functions implement matrix inversion (`blockinv`) and division (`blockmldivide` and `blockmrdivide`) by extracting sub-matrices of a user-defined size and calling the matlab built-ins on them. In certain cases, this means that the built-ins are able to exploit structure in the sub-matrices for very fast inversion and quickly combine the results together.

In general, expect these functions to be slower than simply using built-ins unless you are sure that your sub-matrices (but not the full matrix) have the kind of [structure exploited by mldivide](https://www.mathworks.com/help/matlab/ref/mldivide.html#bt4jslc-6).

The `testBlockFunctions.m` script generates random matrices and asserts that `blockinv`, `blockmldivide`, and `blockmrdivide` are within a reasonable tolerance of their built-in counterparts.

The `profileBlockInv.m` script generates random *structured* matrices of increasing size and profiles the performance of built-in versus `block` functions.
