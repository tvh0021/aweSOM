Distributed MPI streaming
=========================

:mod:`aweSOM.distributed` and :mod:`aweSOM.npy_parts` extend :mod:`aweSOM.stream` across multiple CPU processes with MPI. Each rank handles a subset of the data with its own Numba threads, so training and labelling scale across nodes while keeping memory bounded per rank.

Installation and launch
-----------------------

Install the optional ``[mpi]`` extra to pull in ``mpi4py`` on a system that already provides an MPI library such as OpenMPI or MPICH:

.. code-block:: bash

    pip install ".[mpi]"

Importing :mod:`aweSOM`, :mod:`aweSOM.stream`, :mod:`aweSOM.distributed`, or :mod:`aweSOM.npy_parts` never imports ``mpi4py``. Only :func:`aweSOM.distributed.get_comm` and :func:`aweSOM.distributed.node_local_rank` import ``mpi4py`` lazily when called.

Launch multi-rank scripts through ``python -m mpi4py`` so an unhandled exception on one rank aborts the whole job instead of hanging other ranks in a collective call:

.. code-block:: bash

    mpirun -n 4 python -m mpi4py train_and_label.py

On clusters that use Intel Omni-Path and ``libpsm2``, the fabric library may pin MPI ranks to a single CPU core unless ``HFI_NO_CPUAFFINITY=1`` is exported before launch. That environment variable is a site workaround for ``libpsm2`` that you set in your job script; ``aweSOM`` never sets it.

Communicator helpers
--------------------

:mod:`aweSOM.distributed` exports four public items:

- :func:`aweSOM.distributed.get_comm`: returns ``MPI.COMM_WORLD`` when ``mpi4py`` is installed and more than one rank is running, or ``None`` otherwise. Call it once on the main thread.
- :func:`aweSOM.distributed.row_range`: splits ``n`` rows across ``size`` ranks into deterministic, contiguous ranges ``[start, stop)`` that differ in length by at most one row.
- :func:`aweSOM.distributed.node_local_rank`: returns the zero-based rank of the calling process among the ranks of ``comm`` on the same host (or ``0`` when ``comm`` is ``None``). It uses ``comm.Split_type(MPI.COMM_TYPE_SHARED)``, which is collective across ``comm``, and reads no system information files.
- :class:`aweSOM.distributed.Reducer`: wraps an ``mpi4py`` communicator for rank-0 ``float64`` broadcasts (:meth:`~aweSOM.distributed.Reducer.bcast`), Python object broadcasts (:meth:`~aweSOM.distributed.Reducer.bcast_object`), and rank-ordered ``Allgather`` pass reductions (:meth:`~aweSOM.distributed.Reducer.reduce_pass`).

Multi-rank training, checkpointing, and reproducibility
-------------------------------------------------------

Pass ``comm=get_comm()`` to :func:`aweSOM.stream.train_batch_stream`. Each rank processes the chunks assigned to it round-robin by chunk index, then :class:`aweSOM.distributed.Reducer` exchanges per-rank accumulators with ``Allgather`` and sums them in ascending rank order on every rank. Because every rank sums in the same order, every rank holds bitwise-identical weights after each epoch.

Reproducibility rules:

- **Fixed rank count:** runs with the same ``comm.Get_size()``, ``chunk``, ``groups``, seed, and data source produce bitwise-identical weights.
- **Checkpoint and resume:** only rank 0 writes the checkpoint file and runs ``callback``, then broadcasts the outcome so a stop or error ends every rank cleanly. Resuming with ``checkpoint=..., resume=True`` requires the same rank count as the saved checkpoint and reproduces an uninterrupted run at that rank count bit for bit.
- **Different rank counts:** changing the rank count changes the order in which per-rank ``float64`` accumulators are summed, so trained weights differ in the last bits. The release tests measured a maximum relative weight difference of ``1.3e-13`` with identical labels across 1, 2, and 4 ranks on 40,000 rows and a 24x17 map.

Multi-rank labelling and the one-writer-per-file rule
-----------------------------------------------------

Pass ``comm`` to :func:`aweSOM.stream.predict_stream` so each rank labels only the contiguous slice ``row_range(n_samples, rank, size)`` with no collective communication during labelling.

Each output argument (``out``, ``out_node``, ``out_distance``) may hold either all ``n_samples`` rows or only this rank's ``stop - start`` rows:

- **Local filesystems:** ranks can write non-overlapping slices of one pre-created ``.npy`` file through :class:`aweSOM.stream.NpyWriter` when the filesystem supports concurrent non-overlapping writes cleanly.
- **Shared network filesystems:** keep one writer per file. Have each rank write only its own part file inside a ``<stem>.parts/`` directory so file-level write caching stays enabled and ranks never contend on one inode.

Sharded ``.parts`` layout and ``aweSOM.npy_parts``
--------------------------------------------------

A sharded output directory ``<stem>.parts/`` follows format ``awesom-npy-parts`` version ``1``:

- Each rank ``r`` creates ``part-RRRR.npy`` (name from :func:`aweSOM.npy_parts.part_filename`), writes its ``stop - start`` rows with :class:`aweSOM.stream.NpyWriter`, flushes with ``fsync``, closes the file, and sends its row range, byte count, and CRC-32 to rank 0.
- After every rank closes its part file, rank 0 calls :func:`aweSOM.npy_parts.write_npy_manifest` to validate every part header and atomically publish ``manifest.json`` last. Until ``manifest.json`` appears, readers treat the directory as incomplete.

The primary usage APIs in :mod:`aweSOM.npy_parts` (with :func:`~aweSOM.npy_parts.open_npy`, :class:`~aweSOM.npy_parts.ShardedNpyArray`, and :func:`~aweSOM.npy_parts.stitch_npy` also re-exported from :mod:`aweSOM` and :mod:`aweSOM.stream`) are:

- :func:`aweSOM.npy_parts.parts_dir_for`: maps ``<stem>.npy``, ``<stem>.parts``, or ``<stem>.parts/manifest.json`` to ``Path("<stem>.parts")``.
- :func:`aweSOM.npy_parts.part_filename`: builds the zero-padded part filename (at least four digits, such as ``part-0000.npy``).
- :func:`aweSOM.npy_parts.write_npy_manifest`: validates part headers, byte lengths, and contiguous row coverage, then atomically links ``manifest.json``.
- :func:`aweSOM.npy_parts.open_npy`: opens either a plain ``.npy`` file or a ``<stem>.parts/`` directory. With ``mmap_mode="r"`` (default), it returns a read-only ``numpy.memmap`` for plain ``.npy`` files or a lazy :class:`aweSOM.npy_parts.ShardedNpyArray` for ``.parts`` directories. With ``mmap_mode=None``, it loads an in-memory ``numpy.ndarray`` for either layout.
- :class:`aweSOM.npy_parts.ShardedNpyArray`: lazy read-only array view over the part files using per-part memory maps and ``O(log R)`` binary search on row boundaries. Call ``close()`` when done.
- :func:`aweSOM.npy_parts.verify_npy_parts`: validates ``manifest.json``, every part header and byte size, and every part payload CRC-32.
- :func:`aweSOM.npy_parts.stitch_npy`: checks every part CRC-32 while copying all parts into a single ``.npy`` file that is byte-identical to ``numpy.save``, calls ``fsync`` on the file and parent directory when supported, and removes ``<stem>.parts`` only when ``delete_parts=True``.

See :doc:`modules` for the complete reference of all 15 exports in :mod:`aweSOM.npy_parts`.

Command line interface
----------------------

Verify or stitch a ``<stem>.parts/`` directory from the shell with ``python -m aweSOM.npy_parts``:

.. code-block:: bash

    python -m aweSOM.npy_parts verify labels.parts
    python -m aweSOM.npy_parts stitch labels.parts -o labels.npy
    python -m aweSOM.npy_parts stitch labels.parts --delete-parts

Both commands print a JSON summary to standard output and exit with code 0 on success.

SCE discovery of ``.parts`` outputs
-----------------------------------

:mod:`aweSOM.sce` and :mod:`aweSOM.make_sce_clusters` read both plain ``.npy`` files and ``<stem>.parts/`` directories through :func:`aweSOM.npy_parts.open_npy`. When scanning a folder, ``sce.py`` discovers both ``*.parts/`` directories that contain ``manifest.json`` and ``*.npy`` files in a deterministic order (sorted ``.parts`` stems followed by remaining plain ``.npy`` stems in sorted order). If both ``<stem>.npy`` and ``<stem>.parts/`` exist for the same stem, discovery prefers the plain ``<stem>.npy`` file and skips the duplicate ``.parts`` directory. On the 28-map ``128^3`` plasma example on macOS, plain ``.npy`` SCE runs produce ``multimap_mappings.txt`` and mask files identical by SHA-256 to ``v1.10.1`` and ``v1.9.0``.

End-to-end MPI example
----------------------

Run the script below with ``mpirun -n 2 python -m mpi4py script.py``:

.. code-block:: python

    from pathlib import Path
    import numpy as np
    from aweSOM import Lattice, open_npy, stitch_npy
    from aweSOM.distributed import get_comm, row_range
    from aweSOM.npy_parts import (
        part_filename,
        parts_dir_for,
        verify_npy_parts,
        write_npy_manifest,
    )
    from aweSOM.stream import (
        NpyReader,
        NpyWriter,
        predict_stream,
        train_batch_stream,
    )

    comm = get_comm()
    rank = 0 if comm is None else comm.Get_rank()
    size = 1 if comm is None else comm.Get_size()

    data_path = Path("features.npy")
    reader = NpyReader(str(data_path))
    weights, info = train_batch_stream(
        reader,
        xdim=12,
        ydim=10,
        epochs=10,
        phases=(4, 2, 4),
        chunk=512,
        r_min=2,
        seed=42,
        comm=comm,
    )

    # Cluster on rank 0 and broadcast cluster IDs so every rank labels the same map
    if rank == 0:
        lat = Lattice(xdim=12, ydim=10)
        lat.lattice = weights
        lat.umat = lat.compute_umat()
        node_cluster = lat.assign_cluster_to_lattice(target_clusters=4).ravel(order="F").astype(np.int32)
    else:
        node_cluster = None
    if comm is not None:
        node_cluster = comm.bcast(node_cluster, root=0)

    # Write one part file per rank inside labels.parts/
    parts_dir = parts_dir_for("labels.npy")
    if rank == 0:
        parts_dir.mkdir(exist_ok=True)
    if comm is not None:
        comm.Barrier()

    n_rows = reader.shape[0]
    lo, hi = row_range(n_rows, rank, size)
    part_name = part_filename(rank, size)
    part_path = parts_dir / part_name
    mm = np.lib.format.open_memmap(part_path, mode="w+", dtype=np.uint8, shape=(hi - lo,))
    del mm
    writer = NpyWriter(str(part_path))
    predict_stream(
        weights,
        node_cluster,
        reader,
        out=writer,
        chunk=512,
        backend="patch",
        lattice_shape=(12, 10),
        comm=comm,
    )
    writer.close(sync=True)

    entry = {
        "rank": rank,
        "file": part_name,
        "start": lo,
        "stop": hi,
        "rows": hi - lo,
        "bytes": (hi - lo) * np.dtype(np.uint8).itemsize,
        "crc32": int(writer.crc32),
    }
    entries = [entry] if comm is None else comm.gather(entry, root=0)

    if rank == 0:
        write_npy_manifest(parts_dir, dtype=np.uint8, shape=(n_rows,), parts=entries)
        verify_npy_parts(parts_dir)
        lazy_labels = open_npy(parts_dir)
        try:
            assert lazy_labels.shape == (n_rows,)
        finally:
            lazy_labels.close()
        stitch_npy(parts_dir, "labels.npy")
