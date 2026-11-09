==============================================================================
cLHS: Conditioned Latin Hypercube Sampling
==============================================================================
|docs| |GitHub| |binder| |license| |pepy|

.. |docs| image:: https://readthedocs.org/projects/clhs-py/badge/?version=latest
   :target: https://clhs-py.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |GitHub| image:: https://img.shields.io/badge/GitHub-clhs__py-informational.svg
   :target: https://github.com/wagoner47/clhs_py
   
.. |binder| image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/wagoner47/clhs_py.git/master?filepath=%2Fdocs%2Fnotebooks

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/wagoner47/clhs_py/blob/master/LICENSE.rst
   
.. |pepy| image:: https://static.pepy.tech/personalized-badge/clhs?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads
   :target: https://pepy.tech/project/clhs

This **Python** package is based on the **Conditioned Latin Hypercube Sampling (cLHS)**
method of `Minasny & McBratney (2006)`_. It follows some of the code from the **R** package
clhs_ of Roudier et al.

- It attempts to create a Latin Hypercube sample by selecting only from input data. 
- It uses simulated annealing to force the sampling to converge more rapidly.
- It allows for setting a stopping criterion on the objective function described in `Minasny & McBratney (2006)`_.

You may reproduce the jupyter notebook example on `Binder <https://mybinder.org/v2/gh/wagoner47/clhs_py.git/master?filepath=%2Fdocs%2Fnotebooks>`_.

Please check `online documentation <https://clhs-py.readthedocs.io/en/latest/>`_ for more information.

Installation and development
-----------------------------

Python 3.9 or later is required. Install with ``python -m pip install clhs``.
For development with uv:

.. code-block:: bash

    uv venv --python 3.12
    uv pip install -e ".[test]"
    uv run --no-project python -m pytest

You can also use pip and venv. See ``docs/installation.rst`` for source
installation and build instructions.

Predictor types and resampling
-------------------------------

Floating columns are continuous. Integer and other columns are categorical.
Convert integer measurements to float to treat them as continuous predictors.
Inputs containing only categorical predictors use random swaps with the
categorical objective.

The resampling helpers keep their signatures, but their
``continuous_objective_values`` argument must contain scores aligned with
``sampled_indices``. For each observation, sum the count minus one in its
occupied bin across continuous predictors. The stratum objectives returned by
``continuous_objective_func`` and ``clhs`` are separate quantities and cannot
be passed directly as observation scores. ``clhs`` computes these scores
internally. Corrected scoring can change selected indices compared with older
versions. Calls with the same seed remain reproducible within an environment.


.. _Minasny & McBratney (2006): https://doi.org/10.1016/j.cageo.2005.12.009
.. _clhs: https://CRAN.R-project.org/package=clhs
