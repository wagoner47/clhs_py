About conditioned Latin Hypercube Sampling (cLHS) in Python
===============================================================================

This code is based on the cLHS method of 
`Minasny & McBratney (2006) <https://doi.org/10.1016/j.cageo.2005.12.009>`_.
It follows some of the code from the R package
`clhs <https://cran.r-project.org/web/packages/clhs/>`_ of Roudier et al.

For cLHS the problem is: given N sites with ancillary variables (X), 
select x a sub-sample of size n (n << N) in order that x forms a Latin hypercube, or the multivariate distribution of X is maximally stratified. 

In short, this code attempts to create a Latin Hypercube sample by selecting
only from input data. It uses simulated annealing to force the sampling to
converge more rapidly, and also allows for setting a stopping criterion on
the objective function described in `Minasny & McBratney (2006) <https://doi.org/10.1016/j.cageo.2005.12.009>`_.

Credits: `Erika Wagoner <https://wagoner47.github.io/>`_ (wagoner47) and `Zhonghua Zheng <https://zzheng93.github.io/>`_ (zzheng93)
