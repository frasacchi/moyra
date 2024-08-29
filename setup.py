import shutil
from os import path
from setuptools import find_packages, setup
import os 


dir_path = path.dirname(path.realpath(__file__))

with open(path.join(dir_path,'version.txt'),'r') as f:
    ver = f.read()

# clean up directories
dirs = ['.eggs','build','dist','moyra.egg-info']
for _dir in dirs:
    if path.exists(path.join(dir_path,_dir)):
        shutil.rmtree(path.join(dir_path,_dir))



setup(
    name='moyra',
    packages=find_packages(include=['moyra','moyra.*']),
    version=ver,
    description='Generate Multi-body Symbolic and Numeric Equations of Motion',
    long_description = open(path.join(dir_path,'README.md')).read(),
    long_description_content_type="text/markdown",
    author='Fintan Healy',
    author_email = 'fintan.healy@bristol.ac.uk',
    url='https://github.com/fh9g12/moyra',
    license='MIT',
    install_requires=['sympy','numpy','scipy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)

