from setuptools import find_packages, setup, Extension

# read the contents of README file
from os import path
from io import open  # for Python 2 and 3 compatibility

# # get __version__ from _version.py
# ver_file = path.join('pyhealth', 'version.py')
# with open(ver_file) as f:
#     exec(f.read())

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.rst
def readme():
    with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
        return f.read()


# read the contents of requirements.txt
with open(path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

VERSION = "1.1.4"

setup(
    name="pyhealth",
    version=VERSION,
    description="A Python library for healthcare AI",
    long_description=readme(),
    long_description_content_type="text/x-rst",
    author="Chaoqi Yang, Zhenbang Wu, Patrick Jiang, Zhen Lin, Benjamin Danek, Junyi Gao, Jimeng Sun",
    author_email="chaoqiy2@illinois.edu",
    url="https://github.com/sunlabuiuc/pyhealth",
    keywords=[
        "heathcare AI",
        "healthcare",
        "electronic health records",
        "EHRs",
        "machine learning",
        "data mining",
        "neural networks",
        "deep learning",
    ],
    packages=find_packages(exclude=["test"]),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=["setuptools>=38.6.0"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
    ],
)
