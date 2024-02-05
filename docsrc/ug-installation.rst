Installation
===============


Install the Conda environment
-----------------------------

You may skip this step if your Conda environment has been installed already.

Step 1: Download the installation script for miniconda3
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

macOS (Intel)
'''''''''''''

.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

macOS (Apple Silicon)
'''''''''''''''''''''

.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

Linux
'''''
.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Step 2: Install Miniconda3
"""""""""""""""""""""""""""

.. code-block:: bash

  chmod +x Miniconda3-latest-*.sh && ./Miniconda3-latest-*.sh

During the installation, a path :code:`<base-path>` needs to be specified as the base location of the python environment.
After the installation is done, we need to add the two lines into your shell environment (e.g., :code:`~/.bashrc` or :code:`~/.zshrc`) as below to enable the :code:`conda` package manager (remember to change :code:`<base-path>` with your real location):

.. code-block:: bash

  export PATH="<base-path>/bin:$PATH"
  . <base-path>/etc/profile.d/conda.sh

Step 3: Test your Installation
"""""""""""""""""""""""""""""""

.. code-block:: bash

  source ~/.bashrc  # assume you are using Bash shell
  which python  # should return a path under <base-path>
  which conda  # should return a path under <base-path>


Install `pens`
---------------


Taking a clean installation as example, first let's create a new environment named :code:`pens-env` via :code:`conda`

.. code-block:: bash

    conda create -n pens-env python=3.11
    conda activate pens-env

Then install some dependencies via :code:`conda`:

.. code-block:: bash

    conda install jupyter notebook cartopy

Once the above dependencies have been installed, simply

.. code-block:: bash

    pip install netcdf4 xarray pens
    pip install scipy==1.11.4  # to handle a dependency issue (date: 02-02-2024)

and you are ready to

.. code-block:: python

    import pens

in Python.
