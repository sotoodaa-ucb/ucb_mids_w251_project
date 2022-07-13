#!/usr/bin/env python
import pathlib
import setuptools

here = pathlib.Path(__file__).parent.resolve()

setuptools.setup(
    name='mids_plane_classification',
    packages=setuptools.find_packages(
        include=['mids_plane_classification', 'mids_plane_classification.*']
    ),
)
