## KFS-suite

This is a collection of three Kalman filter and smoother implementations in Matlab:
1) A linear Kalman filter and Rauch-Tung-Striebel smoother
2) A linear Kalman filter and two-filter smoother
3) An iterative nonlinear (quadrature) Kalman filter and Rauch-Tung-Striebel smoother

All three filters/smoothers can correctly treat the following:
- missing data,
- multiple measurements,
- time-varying systems, and
- state constraints (a simple projection-based approach is implemented).

All three filters/smoothers return various quantities that might be of interest for further diagnostics or analyses.
In particular, all three functions return the innovation signals, and the two linear filters return the negative log data likelihood, also known as the **energy**.

Both innovation and energy can be used as optimization targets for **estimating any of the state-space model parameters**, such as the noise covariances.
(Some of the tests/demo cases show how to do this.)

Various tests and demos are included to **verify the correctness** of various aspects of the implementation, and to showcase its utility.
Significant effort has been invested into ensuring the **numerical robustness** and accuracy of the implementation, whereas execution speed was *not* a major concern.

As a rather unusual feature, the two linear filters/smoothers implement the **sample weighting** approach described in Petersen (2022). 
This can be used to perform importance-weighted estimation in the face of model mismatch, covariate shift, and time-varying systems, as described in Petersen (2022) and illustrated in `test_wkfs.m`.

Finally, as a limitation, currently, only the first filter/smoother (linear KF + RTS) supports **inputs** ("+B u(k)"), although the other two could be quite trivially extended to also support this.

Concerning the nonlinear filter and smoother, a **general iterative sigma-point quadrature filter and smoother** has been implemented.
Turning off iterations and choosing the unscented transform as a quadrature scheme, this is equivalent to the classical **unscented Rauch-Tung-Striebel smoother**.
As an alternative quadrature scheme, the general (also multivariate) Gauss-Hermite quadrature rule is provided for use.
Since the implementation is modular, alternative quadrature schemes could easily be integrated.

Concerning the **iterative aspect** of the implementation, it has been shown in a number of studies that approximative nonlinear (e.g., extended, unscented, or Gauss-Hermite) filters and smoothers can greatly benefit from performing iterative filter/smoother runs (see Herzog et al., Tronarp et al. below).
The included demo case (`demo_nonlin.m`)  demonstrates this effect very clearly.
By default, iterations are performed both within individual samples and across whole filter/smoother runs. 
Dampening is implemented to speed up convergence across iterations.

As the documentation is likely far from complete and clear, **please don't hesitate to ask away in case of any questions.**
My wish is for this toolbox to be useful to as many people as possible.
(If there is sufficient demand, I might be willing to put together a dedicated PDF documentation in the future.)
Also, please do not hesitate to contact me in case of any bugs or problems with the code.

Much of the work leading to the development of this toolbox has been done while I was at the [University of Lübeck](https://www.uni-luebeck.de/en/university/university.html), with the [Institute for Electrical Engineering in Medicine](https://www.ime.uni-luebeck.de/institute.html).

#### Third-party content
The following files (in the directory `3rd party code`) have been written by other authors:
- the `block-matrix-inverse-tools` have been written by **Richard Lange**, see https://github.com/wrongu/block-matrix-inverse-tools
- `vline2.m` has been written by **K. Stahl**, see https://www.mathworks.com/matlabcentral/fileexchange/29578-improved-vline
- `hermite_rule.m` has been written by **John Burkardt** and only marginally edited by me, see https://people.math.sc.edu/Burkardt/m_src/hermite_rule/hermite_rule.html
- `RandOrthMat.m` has been written by **Ofek Shilon**, see https://www.mathworks.com/matlabcentral/fileexchange/11783-randorthmat

All of these files are included here for convenience only, and no claim concerning their respective licenses is made.
Copyright remains with the original authors, of course.
Check the original sources (provided above) if in doubt about licensing.


#### References
- Arasaratnam and Haykin (2007), Discrete-Time Nonlinear Filtering Algorithms Using Gauss-Hermite Quadrature, Proceedings of the IEEE.
- Dan Simon (2006), Optimal State Estimation, John Wiley & Sons, Inc.
- Dan Simon (2009), Kalman filtering with state constraints: a survey of linear and nonlinear algorithms, IET Control Theory and Applications.
- Eike Petersen (2022), Model-based Probabilistic Inference for Monitoring Respiratory Effort using the Surface Electromyogram, Dissertation, forthcoming.
- Herzog, Petersen, Rostalski (2019), Iterative Approximate Nonlinear Inference via Gaussian Message Passing on Factor Graphs, IEEE Control Systems Letters.
- Simo Särkkä (2013), Bayesian filtering and smoothing, Cambridge University press.
- Tronarp, Garcia-Fernandez and Särkkä (2018), Iterative Filtering and Smoothing in Nonlinear and Non-Gaussian Systems Using Conditional Moments, IEEE Signal Processing Letters.

Some of the numerical implementation details are my own.


----
Eike Petersen, 2021
