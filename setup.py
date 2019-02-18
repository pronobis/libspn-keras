#!/usr/bin/env python3

from setuptools import setup


def get_readme():
    """Read readme file."""
    with open('README.md') as f:
        return f.read()


###############################
# Setup
###############################
setup(
    ################
    # General Info
    ################
    name='libspn',
    description='The libsak joke in the world',
    long_description=get_readme(),
    setup_requires=[
        'setuptools_scm',  # Use version from SCM using setuptools_scm
        'setuptools_git >= 0.3',  # Ship files tracked by git in src dist
        'colorama',  # For color output
        # For building docs:
        'sphinx',
        'recommonmark',
        'sphinxcontrib-napoleon',
        'sphinxcontrib-websupport',
        'sphinx_rtd_theme',
        # For testing
        'flake8'
    ],
    use_scm_version=True,  # Use version from SCM using setuptools_scm
    classifiers=[
        'Development Status :: 3 - Alpha',
        # 'License :: ', TODO
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=('libspn spn deep-learning deep-learning-library '
              'machine-learning machine-learning-library tensorflow'),
    url='http://www.libspn.org',
    author='Andrzej Pronobis',
    author_email='a.pronobis@gmail.com',
    license='Custom',

    ################
    # Installation
    ################
    packages=['libspn'],
    install_requires=[
        # 'tensorflow', Disabled, it overwrites a GPU installation with manulinux from pypi
        'numpy',
        'scipy',
        'matplotlib',
        'parameterized',
        'pillow',
        'pyyaml',
        'colorama'  # For color output in tests
    ],
    zip_safe=False,
    # Stuff in git repo will be included in source dist
    include_package_data=True,
    ################
    # Tests
    ################
    test_suite='nose2.collector.collector',
    tests_require=['nose2']
)
