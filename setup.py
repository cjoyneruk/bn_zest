import setuptools
import os

ROOT_DIR = os.path.dirname(__file__)
VERSION_FILE = os.path.join(ROOT_DIR, 'bn_zest/_version.py')
VERSION = open(VERSION_FILE, 'r').read().split(' = ')[1:-1]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bn_zest",
    version=VERSION,
    author="C. H. Joyner",
    author_email="c.joyner@qmul.ac.uk",
    description="Lightweight pomegranate wrapper for Bayesian Network construction and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.research.its.qmul.ac.uk/ahw387/bn_zest",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pandas', 'pomegranate'],
    include_package_data=True
)
