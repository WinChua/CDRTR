# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='CDRTR',
    version='0.1.0',
    description='Code for Graduation',
    long_description=readme,
    author='Win Chua',
    author_email='winchua@foxmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

