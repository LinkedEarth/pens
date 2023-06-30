from setuptools import setup, find_packages

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setup(
    name='pens',  # required
    version='2023.6.30',
    description='pens: utilities for comparing paleoclimate reconstruction ensembles',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Feng Zhu, Julien Emile-Geay',
    author_email='fengzhu@ucar.edu, julieneg@usc.edu',
    url='https://github.com/fzhu2e/pens',
    packages=find_packages(),
    include_package_data=True,
    license="GPL 3.0 license",
    zip_safe=False,
    keywords='paleoclimate reconstruction ensembles',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=[
        'termcolor',
        'pandas',
        'tqdm',
        'xarray',
        'dask',
        'stochastic',
        'properscoring',
        'covar'
    ],
    extras_require={
        'pyleo': ['pyleoclim'],
    }
)
