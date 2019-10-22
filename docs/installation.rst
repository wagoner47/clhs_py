Installation instructions
===============================================================================

Currently, the only way to install this package is from source.

#. Clone the github repository::

     git clone https://github.com/wagoner47/clhs_py.git

   Or using SSH clone::

     git clone git@github.com:wagoner47/clhs_py.git

#. Move into the new directory::

     cd clhs_py

#. Run the setup script::

     python setup.py install

You may also supply the `--user` option to install for a single user (which is
helpful if you don't have admin/root privledges, for instance)::

  python setup.py install --user

Other options are also available for the setup script. To see all of them with
documentation, use::

  python setup.py install --help
