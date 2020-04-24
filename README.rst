==============================================================================
cLHS: Conditioned Latin Hypercube Sampling
==============================================================================
|docs| |GitHub| |binder| |license|

.. |docs| image:: https://readthedocs.org/projects/clhs-py/badge/?version=latest
   :target: https://clhs-py.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |GitHub| image:: https://img.shields.io/badge/GitHub-clhs__py-informational.svg
   :target: https://github.com/wagoner47/clhs_py
   
.. |binder| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/wagoner47/clhs_py.git/master?filepath=%2Fdocs%2Fnotebooks

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/wagoner47/clhs_py/blob/master/LICENSE.rst

This **Python** package is based on the **Conditioned Latin Hypercube Sampling (cLHS)**
method of `Minasny & McBratney (2006)`_. It follows some of the code from the **R** package
clhs_ of Roudier et al.

- It attempts to create a Latin Hypercube sample by selecting only from input data. 
- It uses simulated annealing to force the sampling to converge more rapidly.
- It allows for setting a stopping criterion on the objective function described in `Minasny & McBratney (2006)`_.

You may reproduce the jupyter notebook example on `Binder <https://mybinder.org/v2/gh/wagoner47/clhs_py.git/master?filepath=%2Fdocs%2Fnotebooks>`_.

Please check `online documentation <https://clhs-py.readthedocs.io/en/latest/>`_ for more information.


.. _Minasny & McBratney (2006): https://doi.org/10.1016/j.cageo.2005.12.009
.. _clhs: https://CRAN.R-project.org/package=clhs
