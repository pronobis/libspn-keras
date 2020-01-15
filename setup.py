#!/usr/bin/env python3

from setuptools import setup, find_packages


def get_readme():
    """Read readme file."""
    with open('README.md') as f:
        return f.read()


_VERSION = '0.1.3'


REQUIRED_PACKAGES = [
    'tqdm',
    'numpy',
    'scipy',
    'matplotlib',
    'pillow',
    'pyyaml',
]

###############################
# Setup
###############################
setup(
    ################
    # General Info
    ################
    name='libspn',
    version=_VERSION,
    description='LibSPN is a TensorFlow-based library for building and training '
                'Sum-Product Networks.',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    setup_requires=[
        # For building docs:
        'sphinx',
        'recommonmark',
        'sphinxcontrib-napoleon',
        'sphinxcontrib-websupport',
        'sphinx_rtd_theme',
        'm2r'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
    ],
    keywords=('libspn spn deep-learning deep-learning-library '
              'machine-learning machine-learning-library tensorflow'),
    url='http://www.libspn.org',
    author='Andrzej Pronobis, Avinash Raganath, Jos van de Wolfshaar',
    author_email='a.pronobis@gmail.com',
    license='MIT',

    ################
    # Installation
    ################
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
)
