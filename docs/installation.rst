Installation instructions
===============================================================================

cLHS supports Python 3.9 and later. Its runtime requirements are NumPy 1.26.0
or later, pandas 2.0.0 or later, and tqdm 4.66.0 or later. The installer chooses
dependency versions compatible with your Python version.

Install from PyPI
-----------------

.. code-block:: bash

    python -m pip install clhs

Install from source
-------------------

.. code-block:: bash

    git clone https://github.com/wagoner47/clhs_py.git
    cd clhs_py
    python -m pip install .

The project declares setuptools as its build backend in ``pyproject.toml``.
pip creates an isolated build environment and installs the build requirements.

Development with uv
-------------------

uv is optional. To create a local environment and install the tests:

.. code-block:: bash

    uv venv --python 3.12
    uv pip install -e ".[test]"
    uv run --no-project python -m pytest

The ``.venv`` directory is ignored by Git. The library does not require a uv
lockfile.

Development with pip
---------------------

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m pytest

Build distributions
--------------------

.. code-block:: bash

    python -m pip install build
    python -m build

This writes a wheel and a source distribution to ``dist/``. Build and test
tools are separate from the package's runtime dependencies.
