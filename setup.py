"""
Setup script for the eGEGL module

Copyright (c) 2021 Elix, Inc.
"""

from setuptools import setup, find_packages

__version__ = "0.1.0"

install_requires = [
    "numpy",
    "torch",
    "torch-geometric",
    "scipy",
    "pandas",
    "sklearn",
    "networkx",
    "tqdm",
    "selfies",
    "neptune-client",
    "black",
    "isort",
    "mypy",
]

setup(
    name="egegl",
    version=__version__,
    description="eGEGL",
    author="Pierre Wüthrich",
    author_email="pierre.wuthrich@elix-inc.com",
    long_description="Generative library based on GEGL",
    packages=find_packages(),
    install_requires=install_requires,
)
