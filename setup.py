try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import conditioned_latin_hypercube_sampling as clhs

classifiers = [
    "Programming Language :: Python :: 3",
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Astronomy"
    ]

with open("README.rst", "r") as fp:
    long_description = fp.read()

setup(
    name="clhs",
    version=clhs.__version__,
    author=clhs.__author__,
    author_email=clhs.__email__,
    url="https://github.com/wagoner47/clhs_py",
    py_modules=["clhs"],
    description="Conditioned Latin Hypercube Sampling in Python",
    long_description=long_description,
    license="MIT",
    classifiers=classifiers
    )
