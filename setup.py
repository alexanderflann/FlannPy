
from setuptools import setup, find_packages

NAME = 'FlannPy'
LICENCE = '3-Clause BSD'
VERSION = '0.1.0'
AUTHOR = 'Alexander Flann'
DESCRIPTION = 'A Python Library for Mathematical Finance'

setup(
    name = NAME,
    license = LICENCE,
    version = VERSION,
    author = AUTHOR,
    description = DESCRIPTION,
    long_description = open(
        "README.md",
        encoding = "utf-8"
    ).read(),
    long_description_content_type = "text/markdown",
    package_dir = {"": "src"},
    packages = find_packages(where = "src"),
    install_requires=[
        'scipy',
        'numpy'
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)

# python setup.py sdist bdist_wheel
# pip install dist/flannpy-0.1.0-py3-none-any.whl
# twine upload dist/*
