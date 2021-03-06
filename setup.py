#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script for fmu-tools."""
import os
from glob import glob
from os.path import splitext
from os.path import basename, exists
from shutil import rmtree
import fnmatch

from distutils.command.clean import clean as _clean
from setuptools import setup, find_packages

from setuptools_scm import get_version

# ======================================================================================
# Requirements and README
# ======================================================================================


def parse_requirements(filename):
    """Load requirements from a pip requirements file"""
    try:
        lineiter = (line.strip() for line in open(filename))
        return [line for line in lineiter if line and not line.startswith("#")]
    except IOError:
        return []


REQUIREMENTS = parse_requirements("requirements.txt")

SETUP_REQUIREMENTS = [
    "pytest-runner",
    "setuptools >=28",
    "setuptools_scm",
]

TEST_REQUIREMENTS = [
    "pytest",
]


CONSOLE_SCRIPTS = [
    "fmudesign=fmu.tools.sensitivities.fmudesignrunner:main",
    "rmsvolumetrics2csv=fmu.tools.rms.volumetrics:rmsvolumetrics2csv_main",
]


with open("README.md") as readme_file:
    README = readme_file.read()

with open("HISTORY.md") as history_file:
    HISTORY = history_file.read()


# ======================================================================================
# Overriding and extending setup commands; here "clean"
# ======================================================================================


class CleanUp(_clean):
    """Custom implementation of ``clean`` command.
    Overriding clean in order to get rid if "dist" folder and etc
    """

    CLEANFOLDERS = (
        "__pycache__",
        "pip-wheel-metadata",
        ".eggs",
        "dist",
        "build",
        "sdist",
        "wheel",
        ".pytest_cache",
        "docs/apiref",
        "docs/_build",
    )

    CLEANFOLDERSRECURSIVE = ["__pycache__", "_tmp_*", "*.egg-info"]
    CLEANFILESRECURSIVE = ["*.pyc", "*.pyo"]

    @staticmethod
    def ffind(pattern, path):
        """Find files"""
        result = []
        for root, _dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    @staticmethod
    def dfind(pattern, path):
        """Find folders"""
        result = []
        for root, dirs, _files in os.walk(path):
            for name in dirs:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    def run(self):
        """After calling the super class implementation, this function removes
        the directories specific to scikit-build ++."""
        super(CleanUp, self).run()

        for dir_ in CleanUp.CLEANFOLDERS:
            if exists(dir_):
                print("Removing: {}".format(dir_))
            if not self.dry_run and exists(dir_):
                rmtree(dir_)

        for dir_ in CleanUp.CLEANFOLDERSRECURSIVE:
            for pdir in self.dfind(dir_, "."):
                print("Remove folder {}".format(pdir))
                rmtree(pdir)

        for fil_ in CleanUp.CLEANFILESRECURSIVE:
            for pfil in self.ffind(fil_, "."):
                print("Remove file {}".format(pfil))
                os.unlink(pfil)


# ======================================================================================
# Sphinx
# ======================================================================================

CMDSPHINX = {
    "build_sphinx": {
        "project": ("setup.py", "fmu_tools"),
        "version": ("setup.py", get_version()),
        "release": ("setup.py", ""),
        "source_dir": ("setup.py", "docs"),
    }
}

# ======================================================================================
# setup
# ======================================================================================

setup(
    name="fmu_tools",
    use_scm_version={"write_to": "src/fmu/tools/version.py"},
    cmdclass={"clean": CleanUp},
    command_options=CMDSPHINX,
    description="Library for various tools scripts in FMU scope",
    long_description=README + "\n\n" + HISTORY,
    author="Equinor R&T",
    author_email="jriv@equinor.com",
    url="https://github.com/equinor/fmu-tools",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    entry_points={"console_scripts": CONSOLE_SCRIPTS},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    zip_safe=False,
    keywords="fmu, tools",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    test_suite="tests",
    tests_require=TEST_REQUIREMENTS,
    setup_requires=SETUP_REQUIREMENTS,
)
