#!/usr/bin/python
# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.rst", "r") as f:
    long_description = f.read()

setup(
    name='image-analyzer',
    author='QeeqBox',
    author_email='gigaqeeq@gmail.com',
    description="Image analyzer is an interface that simplifies interaction with image-related deep learning models",
    long_description=long_description,
    version='0.1',
    license='AGPL-3.0',
    license_files = ('LICENSE'),
    url='https://github.com/qeeqbox/image-analyzer',
    packages=['imageanalyzer'],
    package_data={'imageanalyzer': ['../data/*']},
    install_requires=[
        'tensorflow',
        'requests',
        'Pillow',
        'bs4',
        'galeodes',
        'keras',
        'opencv-python',
        'numpy',
        'asyncio',
        'aiohttp'
    ],
    python_requires='>=3.5',
    entry_points={
        "console_scripts": [
            'image-analyzer=imageanalyzer.__main__:main_logic'
        ]
    }
)
