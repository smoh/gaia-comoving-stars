#! /usr/bin/env python

# Standard library
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

# Thanks DFM: Handle encoding
major, minor1, minor2, release, serial = sys.version_info
if major >= 3:
    def rd(filename):
        f = open(filename, encoding="utf-8")
        r = f.read()
        f.close()
        return r
else:
    def rd(filename):
        f = open(filename)
        r = f.read()
        f.close()
        return r

setup(
    name="gwb",
    version='v0.1',
    author="Semyeong Oh, Adrian Price-Whelan, David W. Hogg",
    author_email="adrn@princeton.edu",
    packages=["gwb", "gwb.tests"],
    url="https://github.com/adrn/schwimmbad",
    license="MIT",
    description="Gaia wide binaries.",
    long_description=rd("README.md"),
    package_data={"": ["LICENSE"]},
    install_requires=["six"],
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)
