Saving and reusing a trained map
================================

:class:`aweSOM.model.SOMModel` and :class:`aweSOM.model.Scaler` save the
trained ``map`` from the :doc:`Iris example <notebooks/iris>` and its
feature scaling to a file, so later runs can label raw data without ``pickle``.

.. code-block:: python

    from aweSOM import SOMModel, Scaler

    # Fit on the raw training features; kind="manual" matches rs.manual_scaling
    scaler = Scaler.fit(iris.data, kind="manual")

    # from_lattice reads map.features_names, which cannot contain spaces or commas
    map.features_names = [name.replace(" ", "_") for name in feature_names]
    # map already has clusters, so merge_cost is only recorded in the metadata
    model = SOMModel.from_lattice(map, scaler=scaler, merge_cost=merge_threshold)

    # Save to HDF5 or plain text
    model.save("iris_model.h5")
    model.save_text("iris_model.txt")

    # Load either file and label raw, unscaled points
    loaded = SOMModel.load("iris_model.h5")
    loaded_text = SOMModel.load("iris_model.txt")

    raw_new_data = iris.data[:5]
    new_labels = loaded.predict(raw_new_data)
    new_labels, bmu_distances = loaded.predict(raw_new_data, return_distance=True)

``predict`` scales raw input with the stored ``Scaler`` and, with
``return_distance=True``, also returns the Euclidean distance to the best-matching
node in the scaled space. ``backend`` is ``"screen"`` (default), ``"patch"``, or
``"numba"`` (see :doc:`backends`). Files record ``format_version`` 1. Every later
1.x release loads them, and ``load`` refuses a file with a newer version. The text
file can be read without HDF5, for example from other languages.

``Scaler`` works in float64. If you trained the lattice on float32 data scaled by
``rs.manual_scaling``, keep the training mean and two standard deviations in
float32, scale new data with them, and build the model with
``Scaler.identity(n_features)``. The saved file then holds no scaling, so scale every
input this way before ``predict``, never with the new data's own mean and
standard deviation.
