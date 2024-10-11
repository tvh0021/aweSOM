Installation
============

1. Install aweSOM and required dependencies:

    .. code-block:: bash

        git clone https://github.com/tvh0021/aweSOM.git
        cd aweSOM
        pip install .

2. Install JAX with CUDA support separately, otherwise JAX will not recognize the GPU:

    .. code-block:: bash

        pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

If your system does not support CUDA, you can skip this step. SCE will automatically fall back to the CPU. However, the 
CPU-only version can be significantly slower for large datasets (see the `performance tests <testing>`_).

3. Install Sphinx and other required packages for building the documentation (optional):

    .. code-block:: bash

        pip install -r docs/requirements.txt

4. Build the documentation (optional):
    
        .. code-block:: bash
    
            cd docs
            make html
            open build/html/index.html