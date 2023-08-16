from distutils.core import setup
import os
from setuptools import find_packages

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
	# Name of the package 
	name='PyBOP',
	# Packages to include into the distribution 
	packages=find_packages('.'),
	# Start with a small number and increase it with 
	# every change you make https://semver.org 
	version='0.0.1',
	# Chose a license from here: https: // 
	# help.github.com / articles / licensing - a - 
	# repository. For example: MIT 
	license='MIT',
	# Short description of your library 
	description='Python Battery Optimisation and Parameterisation',
	# Long description of your library 
	long_description=long_description,
	long_description_content_type='text/markdown',
	# Either the link to your github or to your website 
	url='https://github.com/pybop-team/PyBOP',
	# List of packages to install with this one 
	install_requires=[
        "pybamm>=23.1",
        "numpy>=1.16",
        "scipy>=1.11",
        "pandas>=2.0",
        "casadi>=3.6",
        "nlopt>=2.6",
	],
	# https://pypi.org/classifiers/ 
	classifiers=[],
    python_requires=">=3.8,<3.12",
)
