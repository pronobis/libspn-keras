#!/usr/bin/env python3

from setuptools import setup


def get_readme():
    """Read readme file."""
    with open('README.rst') as f:
        return f.read()


_VERSION = '0.1'


_packages = {
    'tensorflow': 'tensorflow>=1.12.0',
    'tensorflow-gpu': 'tensorflow-gpu>=1.12.0',
}


EXTRA_PACKAGES = {
    'cpu': [_packages['tensorflow']],
    'gpu': [_packages['tensorflow-gpu']],
}

REQUIRED_PACKAGES = [
    'tqdm',
    'numpy',
    'scipy',
    'matplotlib',
    'parameterized',
    'pillow',
    'pyyaml',
    'colorama'  # For color output in tests
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
        'License :: OSI Approved :: Apache Software License',
    ],
    keywords=('libspn spn deep-learning deep-learning-library '
              'machine-learning machine-learning-library tensorflow'),
    url='http://www.libspn.org',
    author='Andrzej Pronobis, Avinash Raganath, Jos van de Wolfshaar',
    author_email='a.pronobis@gmail.com',
    license='Apache 2.0',

    ################
    # Installation
    ################
    packages=['libspn'],
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    zip_safe=False,
    # Stuff in git repo will be included in source dist
    include_package_data=True,
    ################
    # Tests
    ################
    test_suite='nose2.collector.collector',
    tests_require=['nose2']
)
