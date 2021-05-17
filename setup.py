#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["numpy", "numpy-quaternion", "scipy", "anytree"]

setup(
    author="Peter Hausamann",
    author_email="peter.hausamann@tum.de",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Python utilities for estimating and transforming "
    "rigid body motion.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="rigid_body_motion",
    name="rigid-body-motion",
    packages=find_packages(exclude=["tests"]),
    url="https://github.com/phausamann/rigid-body-motion",
    version="0.6.0",
    zip_safe=False,
)
