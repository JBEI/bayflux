#!/usr/bin/env python

from numpy.distutils.core import Extension, setup

emusFortran = Extension(name = 'emusFortran', sources = ['bayflux/core/emusFortran.f90'])

setup(
    name='bayflux',
    version='1.0',
    description='bayflux ',
    author='The LBNL Quantitative Metabolic Modeling group',
    author_email='tbackman@lbl.gov',
    url='https://github.com/JBEI/bayflux',
    packages=['bayflux'],
    install_requires=['cobra @ git+https://github.com/TylerBackman/cobrapy.git@bayesianSampler-0.3#egg=cobra', 'numpy', 'pandas'],
    license='see license.txt file', 
    keywords = ['metabolism', 'flux'],
    classifiers = [],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    ext_modules = [emusFortran],
    entry_points = {
        'console_scripts': ['bayflux=bayflux.core.commandLine:main'],
    }
    )
