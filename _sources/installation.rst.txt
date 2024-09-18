Installation
============

1. Install aweSOM and required dependencies:

    .. code-block:: bash

        git clone https://github.com/tvh0021/aweSOM.git
        cd aweSOM
        pip install .

2. Install JAX with CUDA support separately:

    .. code-block:: bash

        pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html