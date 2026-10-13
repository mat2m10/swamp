"""
Import setuptools
"""
from setuptools import setup
from setuptools import find_packages

"""
Reading the requirements
"""
with open('requirements.txt') as f:
    content = f.readlines()
    requirements = [x.strip() for x in content]

"""
Setup of the project
"""
setup(name = 'swamp',
      description= "a dimensionality reduction tool for data with population stratification using deep learning techniques",
      install_requires = requirements,
      packages=find_packages(),
      author= "Matthieu de Hemptinne",
      )