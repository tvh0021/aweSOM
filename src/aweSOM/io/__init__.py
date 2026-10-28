"""Readers that turn simulation output into level blocks for the feature pipeline.

``iter_level_blocks(source, variables, iteration)`` yields
:class:`LevelBlock` objects, which unpack as
``(level, block_origin, spacing, {var: ndarray})``:

* :class:`CarpetHDF5Reader`: Carpet (CarpetIOHDF5) 3D output, through
  etkplot when it is importable, else a native h5py reader; blocks and owned
  points match etkplot's ``som-features`` exactly.
* :class:`CarpetXReader`: CarpetX (AsterX) Silo or openPMD-HDF5 output.
* :func:`read_tsv`: CarpetX's 1D TSV lines.

Variables are named as in etkplot (``smallb2``, ``P``, ``rho_b``, ``Bx``,
``vel[0]``, ...); :data:`ASTERX_NAMES` maps them onto AsterX's.
"""

from .blocks import ArrayCache, LevelBlock, iter_level_blocks, open_reader, slabs, sphere_mask
from .carpet import CarpetHDF5Reader
from .carpetx import ASTERX_NAMES, CarpetXReader, read_tsv

__all__ = [
    "LevelBlock",
    "iter_level_blocks",
    "open_reader",
    "CarpetHDF5Reader",
    "CarpetXReader",
    "ASTERX_NAMES",
    "read_tsv",
    "ArrayCache",
    "slabs",
    "sphere_mask",
]
