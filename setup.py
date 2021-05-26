import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bn_zest",
    version="0.3.2",
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
