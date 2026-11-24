CPU streaming
=============

:mod:`aweSOM.stream` trains and labels self-organizing maps chunk by chunk on the CPU for datasets that do not fit in memory. Only the current chunk, one prefetched chunk, and per-node accumulators live in RAM, so disk capacity limits the number of rows rather than system memory.

Batch training does not replace online training. :meth:`aweSOM.som.Lattice.train_lattice` remains the online stochastic trainer, whereas :func:`aweSOM.stream.train_batch_stream` runs the batch SOM update over chunks and produces a different optimization trajectory.

Library defaults and radius schedule
------------------------------------

:func:`aweSOM.stream.train_batch_stream` uses these library defaults:

- ``epochs=60`` with radius phases ``(20, 10, 30)``: 20 epochs at ``r_max``, 10 epochs of linear taper strictly between ``r_max`` and ``r_min``, and 30 epochs at ``r_min``.
- ``r_min=8`` (clamped to ``min(r_min, r_max)``) and ``r_max = max(xdim, ydim) + 1`` when ``r_max`` is omitted.
- ``bmu="auto"``, which times ``"screen"`` and ``"patch"`` on up to 16,384 rows of the first chunk of each pass (``min(16384, len(first_chunk))``) and keeps the faster exact search for the rest of the pass.
- ``final_pass="full"``, which computes the final quantization error and per-node hit counts over all rows after the last epoch.

All arguments after ``epochs`` are keyword-only. Pass ``phases=(P, T, M)`` as three non-negative integers summing to ``epochs`` to override the three phase lengths. When ``phases`` is omitted for any epoch count other than 60, :func:`aweSOM.stream.radius_phases` and :func:`aweSOM.stream.radius_schedule` use the proportion split ``P = ceil(epochs / 2)``, ``T = min(ceil(epochs / 4), epochs - P)``, and ``M = epochs - P - T``, so 40 epochs give ``(20, 10, 10)``. ``info["radius_phases"]`` records the chosen phase tuple, and ``info["radius"]`` holds the per-epoch radius array.

Data source protocol
--------------------

Both :func:`aweSOM.stream.train_batch_stream` and :func:`aweSOM.stream.predict_stream` accept two kinds of ``data_source`` containing already scaled, finite rows of shape ``(N, F)``:

1. **Row-sliceable 2D sources.** Any object with a 2D ``shape`` that supports contiguous row slicing ``data[start:stop]``, including a NumPy ``ndarray``, a read-only ``numpy.memmap``, an ``h5py.Dataset``, or :class:`aweSOM.stream.NpyReader`. :class:`aweSOM.stream.NpyReader` reads slices from a C-order 2D ``.npy`` file with plain file reads instead of memory mapping, so touched file pages do not grow process RSS.
2. **Zero-argument callables.** A callable ``fn()`` that returns a fresh iterable of ``(rows, F)`` chunks on each call. On :func:`aweSOM.stream.train_batch_stream`, pass ``n_samples`` and ``n_features`` (or set ``fn.n_feat`` or ``fn.n_features``) to skip the initial counting pass. ``n_features`` is a training-only keyword argument: :func:`aweSOM.stream.predict_stream` takes ``n_samples`` and reads the feature count from ``fn.n_feat`` or ``fn.n_features`` on the callable, so set ``fn.n_feat`` (or ``fn.n_features``) alongside ``n_samples`` when skipping the counting pass during prediction. Optional attributes on the callable extend this protocol:

   - ``fn.rows(ix)`` returns the rows at indices ``ix`` directly for ``init="sampling"`` instead of scanning the stream.
   - ``fn.lazy_chunks()`` yields ``(n_rows, read)`` pairs where ``read()`` loads a chunk only when needed.
   - ``fn.calls`` is an integer counter that checkpoints save and restore on resume so stateful callables reproduce the same sequence of passes. When the counting pass is skipped, ``aweSOM.stream`` increments ``fn.calls`` by one in its place.

``float32`` and ``float64`` chunks stay in their stored dtype during search and accumulate in ``float64``. Other dtypes convert to ``float64`` chunk by chunk.

Chunking, threads, and reproducibility
--------------------------------------

:func:`aweSOM.stream.train_batch_stream` splits each chunk into ``groups`` contiguous row ranges (64 by default), accumulates per-node sums into separate ``float64`` slots, and reduces the slots in a fixed order at the end of the epoch. For a fixed ``chunk`` and ``groups``, trained weights are bitwise independent of the Numba thread count and of whether ``bmu`` is ``"auto"``, ``"screen"``, ``"patch"``, or ``"numba"``. Changing ``chunk`` or ``groups`` changes ``float64`` summation order in the last bits, so bitwise reproducibility requires the same ``chunk`` and ``groups``.

Peak memory during training is roughly ``2 * chunk * F * itemsize + chunk * 12`` bytes for the active chunk, prefetched chunk, and BMU buffers, plus ``groups * nodes * (F + 1) * 8`` bytes for the ``float64`` accumulators.

:func:`aweSOM.stream.predict_stream` evaluates each row independently, so its cluster labels, BMU node indices, and Euclidean distances are bitwise identical across ``chunk`` sizes and across the ``"screen"``, ``"patch"``, and ``"numba"`` backends.

In single-process runs, ``info`` from :func:`aweSOM.stream.train_batch_stream` and ``stats`` from :func:`aweSOM.stream.predict_stream` report ``ranks = 1`` and ``rank = 0``. :func:`aweSOM.stream.peak_rss_bytes` and ``info["peak_rss"]`` return the process peak resident set size in bytes on POSIX systems and ``None`` when Python's ``resource`` module is absent.

Multi-rank MPI streaming
------------------------

Both :func:`aweSOM.stream.train_batch_stream` and :func:`aweSOM.stream.predict_stream` accept an optional ``comm`` keyword argument holding an ``mpi4py`` communicator (such as ``comm=aweSOM.distributed.get_comm()``):

- In :func:`aweSOM.stream.train_batch_stream`, each rank processes chunks round-robin by chunk index, and :class:`aweSOM.distributed.Reducer` exchanges per-rank accumulators with ``Allgather`` and sums them in rank order so every rank holds bitwise-identical weights. Runs with a fixed rank count are bitwise reproducible (including checkpoint and resume, where the rank count must match the checkpoint). Across different rank counts, the ``float64`` sums are added in a different order, so trained weights differ in the last bits (release tests on 40,000 rows and a 24x17 map measured a maximum relative weight difference of ``1.3e-13`` with identical labels across 1, 2, and 4 ranks).
- In :func:`aweSOM.stream.predict_stream`, each rank labels only its contiguous slice ``aweSOM.distributed.row_range(n_samples, rank, size)`` without collectives during labelling. Output targets (``out``, ``out_node``, ``out_distance``) may hold either all ``n_samples`` rows or only this rank's ``stop - start`` rows, allowing each rank to write an independent part file inside ``<stem>.parts/``.

See :doc:`distributed` for installation of the ``[mpi]`` extra, the one-writer-per-file rule on shared network filesystems, sharded ``.parts`` outputs in :mod:`aweSOM.npy_parts`, and a full MPI example.

Checkpoint and resume
---------------------

Pass ``checkpoint="run.npz"`` to :func:`aweSOM.stream.train_batch_stream` to atomically rewrite a checkpoint file after every epoch in format ``aweSOM.stream.checkpoint/1``. Inspect a saved checkpoint with :func:`aweSOM.stream.read_checkpoint`, which returns ``(meta, arrays)``.

Pass ``checkpoint="run.npz", resume=True`` to continue from the last completed epoch. If the checkpoint file does not exist yet, training starts from epoch 0. On resume, the trainer checks that the lattice shape, feature count, sample count, epoch count, radius schedule (including ``r_max``, ``r_min``, and ``phases``), sampling settings, ``chunk``, ``groups``, and rank count match the checkpoint, raising ``ValueError`` if any setting differs. A resumed run produces weights, hit counts, and quantization errors bitwise equal to an uninterrupted run at the same rank count. Under MPI, rank 0 writes the checkpoint and invokes ``callback``, then broadcasts the outcome so a stop or error ends every rank cleanly.

Streaming inference, narrow label dtypes, and ``out_node``
----------------------------------------------------------

:func:`aweSOM.stream.predict_stream` labels a streamed data source using ``backend="screen"`` (default), ``"patch"`` (with ``lattice_shape=(xdim, ydim)``), or ``"numba"``:

- When ``out=None``, :func:`aweSOM.stream.predict_stream` allocates an in-memory array using :func:`aweSOM.inference.narrow_label_dtype`: ``uint8`` when all cluster IDs are non-negative and at most 255, ``uint16`` when non-negative and at most 65,535, and ``int32`` otherwise (including negative IDs or empty arrays). Pass a preallocated ``int32`` ``out`` array, or cast the returned array, when you need signed arithmetic on the labels. :meth:`aweSOM.som.Lattice.predict` continues to return ``int32``. :func:`aweSOM.run_som.save_cluster_labels` uses the same narrowing helper for integer arrays while preserving ``v1.4.0`` file bytes for all integer, float, and boolean inputs.
- Pass ``out_node`` to write the BMU node index for each row alongside cluster labels (for example into a ``uint16`` or ``int32`` array or :class:`aweSOM.stream.NpyWriter`). Pass ``out_distance`` to write the ``float64`` Euclidean BMU distance for each row.
- :class:`aweSOM.stream.NpyWriter` writes contiguous row slices directly to a precreated C-order ``.npy`` file with ``os.pwrite``, tracking a running zlib CRC-32 for ascending contiguous writes that matches :func:`aweSOM.stream.npy_crc32`. Always call ``writer.close()`` before reading the file. :func:`aweSOM.stream.copy_npy_rows` copies a part ``.npy`` file into an open :class:`aweSOM.stream.NpyWriter` and returns the copied CRC-32.

Quality metrics
---------------

Exact ``float64`` quantization error and topographic error live in :mod:`aweSOM._batch_core`:

- ``quantization_error(weights, data)`` and ``topographic_error(weights, data, xdim, ydim)`` compute exact ``float64`` direct differences in bounded memory blocks, breaking ties by lowest node index.
- ``quantization_error_fast(weights, data)`` and ``topographic_error_fast(weights, data, xdim, ydim)`` run a compiled multithreaded Numba scan for finite inputs and match the direct reference within ``float64`` rounding.

Example
-------

.. code-block:: python

    import tempfile
    from pathlib import Path
    import numpy as np
    from aweSOM import Lattice, Scaler
    from aweSOM.stream import (
        NpyReader,
        NpyWriter,
        npy_crc32,
        predict_stream,
        read_checkpoint,
        train_batch_stream,
    )

    rng = np.random.default_rng(42)
    raw = rng.normal(size=(2_000, 4)).astype(np.float32)
    scaler = Scaler.fit(raw, kind="manual")
    scaled = scaler.transform(raw).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        data_path = tmp_dir / "features.npy"
        np.save(data_path, scaled)

        # Train from disk with NpyReader and per-epoch checkpointing
        reader = NpyReader(str(data_path))
        ckpt = str(tmp_dir / "som_ckpt.npz")
        weights, info = train_batch_stream(
            reader,
            xdim=12,
            ydim=10,
            epochs=10,
            phases=(4, 2, 4),
            chunk=512,
            r_min=2,
            seed=42,
            checkpoint=ckpt,
            resume=True,
        )
        meta, _ = read_checkpoint(ckpt)
        assert meta["epoch"] == 10 and info["ranks"] == 1

        # Cluster the trained lattice with Lattice, return labels in memory (out=None),
        # and write BMU node IDs to disk via NpyWriter
        lat = Lattice(xdim=12, ydim=10)
        lat.lattice = weights
        lat.umat = lat.compute_umat()
        clusters = lat.assign_cluster_to_lattice(target_clusters=4)
        node_cluster = clusters.ravel(order="F")

        nodes_path = tmp_dir / "bmu_nodes.npy"
        mm = np.lib.format.open_memmap(nodes_path, mode="w+", dtype=np.uint16, shape=(len(scaled),))
        del mm
        node_writer = NpyWriter(str(nodes_path))
        dist = np.empty(len(scaled), dtype=np.float64)

        labels, stats = predict_stream(
            weights,
            node_cluster,
            reader,
            out=None,
            out_distance=dist,
            out_node=node_writer,
            chunk=512,
            backend="patch",
            lattice_shape=(12, 10),
        )
        node_writer.close()

        assert labels.dtype == np.uint8
        assert node_writer.crc32 == npy_crc32(str(nodes_path), 0, len(scaled))
        assert stats["rows"] == len(scaled)

        # Predict from a zero-argument callable carrying its feature count (n_feat)
        # so predict_stream skips the counting pass with n_samples set
        def stream_chunks():
            stream_chunks.passes += 1
            for start in range(0, len(scaled), 512):
                yield scaled[start : start + 512]

        stream_chunks.passes = 0
        stream_chunks.n_feat = scaled.shape[1]

        callable_labels, callable_stats = predict_stream(
            weights,
            node_cluster,
            stream_chunks,
            out=None,
            n_samples=len(scaled),
            chunk=512,
            backend="patch",
            lattice_shape=(12, 10),
        )
        assert stream_chunks.passes == 1
        assert np.array_equal(callable_labels, labels)
        assert callable_stats["rows"] == len(scaled)
