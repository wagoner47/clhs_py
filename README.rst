==============================================================================
cLHS: Conditioned Latin Hypercube Sampling
==============================================================================

.. image:: https://readthedocs.org/projects/clhs-py/badge/?version=latest
   :target: https://clhs-py.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. module:: clhs

.. moduleauthor:: Erika Wagoner <wagoner47+clhs@email.arizona.edu>
.. sectionauthor:: Erika Wagoner <wagoner47+clhs@email.arizona.edu>

.. include-marker-do-not-remove

.. image:: https://img.shields.io/badge/GitHub-clhs__py-informational.svg
   :target: https://github.com/wagoner47/clhs_py

.. image:: https://img.shields.io/github/license/wagoner47/clhs_py.svg
   :target: https://github.com/wagoner47/clhs_py/blob/master/LICENSE.rst

Conditioned Latin Hypercube Sampling in Python.

This code is based on the conditioned LHS method of
`Minasny & McBratney (2006)`_. It follows some of the code from the R package
clhs_ of Roudier et al.

In short, this code attempts to create a Latin Hypercube sample by selecting
only from input data. It uses simulated annealing to force the sampling to
converge more rapidly, and also allows for setting a stopping criterion on
the objective function described in Minasny & McBratney (2006).



.. _Minasny & McBratney (2006): https://doi.org/10.1016/j.cageo.2005.12.009
.. _clhs: https://CRAN.R-project.org/package=clhs
