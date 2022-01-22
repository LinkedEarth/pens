from setuptools import setup, find_packages

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setup(
    name='pens',  # required
    version='0.0.3',
    description='pens: utilities for comparing paleoclimate reconstruction ensembles',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Feng Zhu',
    author_email='fzhu@nuist.edu.cn',
    url='https://github.com/fzhu2e/pens',
    packages=find_packages(),
    include_package_data=True,
    license="GPL 3.0 license",
    zip_safe=False,
    keywords='paleocliamte reconstruction ensembles',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=[
        'termcolor',
        'pandas',
        'tqdm',
        'xarray',
        'dask',
    ],
)
