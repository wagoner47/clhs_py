Installation instructions
===============================================================================


Install on local machine with ``pip``
-------------------------------------

.. code-block:: bash

    $ pip install clhs


Install on local machine from source
------------------------------------

The get the latest verson that is not uploaded to PyPI yet:

#. Clone the github repository

   .. code-block:: bash

      $ git clone https://github.com/wagoner47/clhs_py.git


   Or using SSH clone

   .. code-block:: bash

      $ git clone git@github.com:wagoner47/clhs_py.git

#. Move into the new directory

   .. code-block:: bash

      $ cd clhs_py

#. Run the setup script

   .. code-block:: bash

      $ python setup.py install

You may also supply the `--user` option to install for a single user (which is
helpful if you don't have admin/root privledges, for instance)

.. code-block:: bash

   $ python setup.py install --user


Other options are also available for the setup script. To see all of them with
documentation, use

.. code-block:: bash

   $ python setup.py install --help
