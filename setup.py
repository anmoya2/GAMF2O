
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    "numpy>=1.19.5",
    "scikit-learn>=1.1"
]

extras_require = {}

setup(
    #  Project name.
    #  $ pip install gamf2o
    name='gamf2o',

    # Version
    version='1.0',

    # Description
    description='Library for Multi-fidelity Optimization using Genetic Algorithms and Deep Learning',

    # Long description (README)
    long_description=long_description,

    # URL
    url='https://github.com/anmoya2/GAMF2O',

    # Author
    author='Antonio R. Moya Martín-Castaño',

    # Author email
    author_email='amoya@uco.es',

    # Keywords
    keywords=['Deep Learning',
              'Multi-fidelity Optimization',
              'Evolutionary Algorithms',
              'Hyperparameter Optimization'
              ],

    # Packages # Excluimos las carpetas del código que no se usen en la librería
    package_dir={"": "./"},
    packages=find_packages(where="./", exclude=['docs']),
    include_package_data=True,
    # Requeriments
    install_requires=install_requires,
    extras_require=extras_require,
    long_description_content_type='text/markdown'
)
