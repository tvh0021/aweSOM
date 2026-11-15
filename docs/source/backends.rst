Inference backends
==================

:mod:`aweSOM.inference` provides three CPU best-matching-unit backends that return bitwise-identical ``int32`` node indices and ``float64`` distances:

- ``"screen"`` is the default since ``v1.4.0`` and calls :func:`aweSOM.inference.best_match_nodes_screen`. It scans all nodes in ``float32`` across 512-point tiles in parallel Numba threads, checks whether the ``float32`` winner stays strictly ahead of the runner-up across the ``float32`` rounding bound in ``float64``, recomputes a certified winner's squared distance in ``float64``, and falls back to the ascending ``float64`` scan on uncertified rows or when the ``float32`` runner-up squared distance overflows to ``inf``.
- ``"patch"``, added in ``v1.9.0``, calls :func:`aweSOM.inference.best_match_nodes_patch`. It cuts the ``(xdim, ydim)`` lattice into ``side x side`` rectangular patches, where ``default_patch_side`` picks ``side=4`` up to ``8,000`` nodes and ``side=5`` above ``8,000`` nodes. For each query point, it computes ``float64`` squared distances to all patch centroids, scans the nearest-centroid patch first, skips patches whose bounding-ball triangle-inequality lower bound exceeds the current ``float64`` best distance, and scans the remaining patches in ``float64``. It returns exact ``float64`` winners and distances with zero ``float32`` rounding ambiguity.
- ``"numba"`` calls :func:`aweSOM.inference.best_match_nodes`, the exhaustive ascending ``float64`` scan over every node in parallel Numba threads. It is the reference kernel for exactness checks.

When to pick patch or screen
----------------------------

``"screen"`` stays the default. Benchmark medians below come from ``aweSOM.predict`` on a 10-core Apple M1 Max at 1-minute load below ``2.0``:

- **Small trained lattices.** On maps of a few hundred nodes such as ``29x20`` with ``580`` nodes, ``"screen"`` is fastest at ``0.00691 s`` versus ``0.0145 s`` for ``"patch"`` across ``100,000`` points, about ``2.1x`` faster, because scanning ``580`` nodes in ``float32`` tiles costs less than computing patch-centroid distances and tracking patch bounds.
- **Medium trained lattices.** Around ``1,800`` to ``2,500`` nodes, such as ``51x36`` with ``1,836`` nodes or the ``64x37`` plasma map with ``2,368`` nodes, ``"patch"`` and ``"screen"`` stay within ``25`` to ``50`` percent, or ``1.3x`` to ``1.5x``. Across ``1,000,000`` points, ``"patch"`` takes ``0.121 s`` versus ``0.160 s`` for ``"screen"`` on ``51x36`` and ``0.119 s`` versus ``0.177 s`` on ``plasma``, while running ``2.6x`` to ``3.5x`` faster than ``"numba"``.
- **Large smooth trained lattices.** At several thousand nodes or more, such as ``91x64`` with ``5,824`` nodes or ``100x100`` with ``10,000`` nodes, ``"patch"`` evaluates ``218`` of ``10,000`` nodes per point on ``100x100`` and finishes ``1,000,000`` points in ``0.298 s`` versus ``0.827 s`` for ``"screen"`` and ``1.69 s`` for ``"numba"``, ``2.8x`` and ``5.7x`` faster.
- **Untrained or non-smooth weights.** ``"patch"`` relies on spatial smoothness across adjacent nodes of a trained SOM so patch radii stay small. On random untrained weights, patch bounding balls overlap and the skip test stays near ``1.0x`` pruning, making ``"patch"`` ``7x`` to ``8x`` slower than ``"screen"`` at ``0.146 s`` versus ``0.0194 s`` on ``51x36`` across ``100,000`` points, although winners and distances remain bitwise exact.

Callers and lattice shape
-------------------------

- :meth:`aweSOM.model.SOMModel.predict` accepts ``backend="screen"`` by default, ``"patch"``, or ``"numba"``, and forwards ``lattice_shape=(self.xdim, self.ydim)`` automatically.
- :func:`aweSOM.inference.predict` accepts ``backend="patch"`` when the caller passes ``lattice_shape=(xdim, ydim)`` with ``xdim * ydim == weights.shape[0]``. The ``weights`` array has C-contiguous shape ``(xdim * ydim, F)`` with node index ``x + y * xdim``, the same layout as ``Lattice.lattice`` and ``SOMModel.weights``. Calling ``aweSOM.predict(..., backend="patch")`` without ``lattice_shape`` raises ``ValueError``. Do not swap ``(xdim, ydim)`` into ``(ydim, xdim)`` when calling :func:`aweSOM.inference.best_match_nodes_patch` or ``aweSOM.predict(..., backend="patch", lattice_shape=(xdim, ydim))``. A swapped shape has the same product ``xdim * ydim`` and still returns exact winners and distances, but it groups non-adjacent nodes into each patch and inflates patch radii and node evaluations, rising from ``87`` to ``1,516`` evaluations per point, about ``17x`` more work, on ``51x36``. The ``node_cluster`` array passed to :func:`aweSOM.inference.predict` is 1D of length ``xdim * ydim``. If you start from ``som.lattice_assigned_clusters`` of shape ``(xdim, ydim)``, flatten it in Fortran order with ``clusters.ravel(order="F")``, as :meth:`aweSOM.som.Lattice.predict` and :meth:`aweSOM.model.SOMModel.from_lattice` do.
- :meth:`aweSOM.som.Lattice.predict` and :meth:`aweSOM.som.Lattice.best_match` accept ``"screen"``, ``"numba"``, and ``"legacy"``, not ``"patch"``. To run ``"patch"`` on a trained ``Lattice``, build a :class:`aweSOM.model.SOMModel` with :meth:`aweSOM.model.SOMModel.from_lattice` or call :func:`aweSOM.inference.predict` with ``lattice_shape=(som.xdim, som.ydim)``.

.. code-block:: python

    import numpy as np
    import aweSOM
    from aweSOM import Lattice, SOMModel, Scaler
    import aweSOM.run_som as rs

    rng = np.random.default_rng(0)
    raw_data = rng.normal(size=(200, 4))
    scaler = Scaler.fit(raw_data, kind="manual")
    scaled_data = rs.manual_scaling(raw_data)

    xdim, ydim = 10, 8
    som = Lattice(xdim=xdim, ydim=ydim, alpha_0=0.5, train=2000)
    som.train_lattice(scaled_data, ["f0", "f1", "f2", "f3"], history=False)
    clusters = som.assign_cluster_to_lattice(merge_cost=0.2)

    # SOMModel forwards lattice_shape=(xdim, ydim) automatically
    model = SOMModel.from_lattice(som, scaler=scaler, merge_cost=0.2)
    labels_model = model.predict(raw_data, backend="patch")

    # aweSOM.predict on scaled data requires explicit lattice_shape=(xdim, ydim)
    node_cluster = clusters.ravel(order="F")
    labels_direct = aweSOM.predict(
        som.lattice,
        node_cluster,
        scaled_data,
        backend="patch",
        lattice_shape=(xdim, ydim),
    )

Tie-breaking and non-finite inputs
----------------------------------

:func:`aweSOM.inference.best_match_nodes`, :func:`aweSOM.inference.best_match_nodes_screen`, and :func:`aweSOM.inference.best_match_nodes_patch` share the same tie-breaking and non-finite contract:

- **Exact ties.** When two or more nodes achieve the same minimum ``float64`` squared Euclidean distance, all three functions return the lowest node index, and :func:`aweSOM.inference.predict` assigns ``node_cluster[lowest_idx]``. ``best_match_nodes`` scans nodes in ascending order and updates the winner only when ``d < best``. ``best_match_nodes_patch`` retains every patch whose bounding ball can hold an equal distance and compares candidates lexicographically by ``(distance, index)``. ``best_match_nodes_screen`` includes input and sum rounding in its ``float32`` error bound, so every exact ``float64`` tie falls inside the rounding interval and runs the ascending ``float64`` fallback scan.
- **NaN and infinite query rows.** Any candidate squared distance that evaluates to ``NaN``, such as a query row with a ``NaN`` coordinate or ``inf - inf``, fails ``d < best`` and never wins. When no node yields a finite ``float64`` squared distance, such as a query row with a ``NaN``, ``+inf``, or ``-inf`` coordinate against finite weights, all three functions return node index ``0`` and squared distance ``inf``. In that case, :func:`aweSOM.inference.predict` assigns ``node_cluster[0]`` and, with ``return_distance=True``, returns Euclidean distance ``inf``.
- **Non-finite and extreme weights.** ``best_match_nodes`` skips any node whose squared distance is ``NaN`` or ``inf``, so one corrupt node never masks finite nodes, and returns node ``0`` and squared distance ``inf`` when every node is non-finite. ``best_match_nodes_screen`` and ``best_match_nodes_patch`` check ``weights`` up front and call ``best_match_nodes`` directly whenever ``weights`` has a non-finite entry or ``max(|weights|) > 1e15``.
- **Runner-up overflow in float32.** Since ``v1.9.0``, ``best_match_nodes_screen`` checks ``s2[p] < inf`` before trusting the ``float32`` margin. Finite ``float64`` inputs whose runner-up squared distance overflows ``float32`` above ``~3.4e38``, along with single-node lattices where ``s2`` stays ``inf``, fall back to the exact ``float64`` scan instead of accepting a ``float32`` winner prematurely.
