"""Frozen, pickle-free SOM model artifact.

A :class:`SOMModel` holds everything needed to classify new data with a
trained lattice: the scaler fitted on the training sample, the node weights,
the node-to-cluster map, distance thresholds for flagging unfamiliar points,
and provenance metadata. It is written either to HDF5 (``save``) or to a
self-describing plain-text file (``save_text``) that the header-only C++
classifier in ``cpp/awesom_classify.hpp`` reads.

Text format (one model per file)::

    # format awesom-som-text
    # xdim 10
    # ydim 8
    # n_features 4
    # reject_distance 0.123...
    # scaler_kind manual
    # bulk_range 1
    # <metadata key> <value, may contain spaces>
    # feature_names a,b,c,d
    # scaler_offset v v v v
    # scaler_scale v v v v
    # node_dist_q99 v v ... (one per node, nan for empty nodes)
    w_0 w_1 ... w_{F-1} cluster      (one line per node, node index = x + y * xdim)

Floats are written with ``%.17g`` so that they read back bitwise.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import io
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .inference import DEFAULT_CHUNK, best_match_nodes
from .inference import predict as _predict

if TYPE_CHECKING:
    from .som import Lattice

__all__ = ["Scaler", "SOMModel"]

FORMAT_VERSION = "1"
_TEXT_MAGIC = "awesom-som-text"
_SCALER_KINDS = ("none", "manual", "minmax")
# header keys with a fixed meaning; metadata keys may not reuse them
_RESERVED_KEYS = frozenset(
    {
        "format",
        "xdim",
        "ydim",
        "n_features",
        "reject_distance",
        "scaler_kind",
        "bulk_range",
        "feature_names",
        "scaler_offset",
        "scaler_scale",
        "node_dist_q99",
    }
)


def _fmt(x: float) -> str:
    return "%.17g" % x


@dataclass
class Scaler:
    """Per-feature affine scaling, ``scaled = (x - offset) / scale``.

    Attributes:
        kind (str): "none", "manual" (mean and two standard deviations, as
            ``run_som.manual_scaling``) or "minmax".
        offset (np.ndarray): float64 [F] value subtracted from each feature.
        scale (np.ndarray): float64 [F] divisor of each feature.
        bulk_range (float): The ``bulk_range`` used for "manual" scaling; it is
            already folded into ``scale`` and kept for provenance only.
    """

    kind: str
    offset: np.ndarray
    scale: np.ndarray
    bulk_range: float = 1.0

    def __post_init__(self):
        if self.kind not in _SCALER_KINDS:
            raise ValueError(f"scaler kind must be one of {_SCALER_KINDS}, got {self.kind!r}")
        self.offset = np.array(self.offset, dtype=np.float64, copy=True).ravel()
        self.scale = np.array(self.scale, dtype=np.float64, copy=True).ravel()
        if self.offset.shape != self.scale.shape:
            raise ValueError("offset and scale must have the same length")
        bad = ~np.isfinite(self.scale) | (self.scale == 0.0)
        if np.any(bad):
            warnings.warn(
                f"scaler scale is zero or non-finite for features {np.flatnonzero(bad).tolist()}; "
                "using 1.0 for those features",
                RuntimeWarning,
                stacklevel=3,
            )
            self.scale[bad] = 1.0
        self.bulk_range = float(self.bulk_range)

    @classmethod
    def identity(cls, n_features: int) -> "Scaler":
        """Return a scaler that leaves the data unchanged.

        Args:
            n_features (int): Number of features.

        Returns:
            Scaler: kind "none" with offset 0 and scale 1.
        """
        return cls("none", np.zeros(n_features), np.ones(n_features))

    @classmethod
    def fit(cls, data: np.ndarray, kind: str = "manual", bulk_range: float = 1.0) -> "Scaler":
        """Fit a scaler on a training sample.

        For "manual", ``offset = mean`` and ``scale = 2 * std / bulk_range``, so
        ``transform`` equals ``run_som.manual_scaling(data, bulk_range)`` bitwise
        when ``bulk_range`` is a power of two (including the default 1.0) and to
        within one rounding otherwise. For "minmax", ``offset = min`` and
        ``scale = max - min``.

        Args:
            data (np.ndarray): [N, F] training sample.
            kind (str, optional): "none", "manual" or "minmax". Defaults to "manual".
            bulk_range (float, optional): Range that holds ~95% of the scaled
                data ("manual" only). Defaults to 1.0.

        Returns:
            Scaler: the fitted scaler.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array")
        n_features = data.shape[1]
        if kind == "none":
            return cls.identity(n_features)
        if kind == "manual":
            offset = np.mean(data, axis=0)
            scale = 2.0 * np.std(data, axis=0) / bulk_range
            return cls("manual", offset, scale, bulk_range)
        if kind == "minmax":
            lo = np.min(data, axis=0)
            return cls("minmax", lo, np.max(data, axis=0) - lo)
        raise ValueError(f"scaler kind must be one of {_SCALER_KINDS}, got {kind!r}")

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Scale raw data.

        Args:
            x (np.ndarray): [N, F] or [F] raw data.

        Returns:
            np.ndarray: float64 ``(x - offset) / scale``.
        """
        return (np.asarray(x) - self.offset) / self.scale

    def inverse_transform(self, x_scaled: np.ndarray) -> np.ndarray:
        """Undo ``transform`` (up to rounding).

        Args:
            x_scaled (np.ndarray): [N, F] or [F] scaled data.

        Returns:
            np.ndarray: float64 ``x_scaled * scale + offset``.
        """
        return np.asarray(x_scaled) * self.scale + self.offset


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _default_metadata() -> dict[str, str]:
    from . import __version__

    return {
        "format_version": FORMAT_VERSION,
        "awesom_version": __version__,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }


def _check_metadata(metadata: dict) -> dict[str, str]:
    clean = {}
    for key, value in metadata.items():
        key, value = str(key), str(value)
        if not key or any(c.isspace() for c in key):
            raise ValueError(f"metadata key {key!r} must be non-empty and contain no whitespace")
        if key in _RESERVED_KEYS:
            raise ValueError(f"metadata key {key!r} is reserved")
        if "\n" in value or "\r" in value:
            raise ValueError(f"metadata value for {key!r} must be a single line")
        clean[key] = value
    return clean


def distance_quantiles(
    nodes_idx: np.ndarray, dist: np.ndarray, n_nodes: int, q_node: float = 99.0, q_all: float = 99.9
) -> tuple[np.ndarray, float]:
    """Per-node and global quantiles of the BMU distance.

    Args:
        nodes_idx (np.ndarray): [N] BMU index of each point.
        dist (np.ndarray): [N] Euclidean BMU distance of each point.
        n_nodes (int): Number of nodes.
        q_node (float, optional): Per-node percentile. Defaults to 99.0.
        q_all (float, optional): Global percentile. Defaults to 99.9.

    Returns:
        tuple[np.ndarray, float]: float64 [n_nodes] per-node percentile (NaN for
            nodes with no points) and the global percentile.
    """
    node_q = np.full(n_nodes, np.nan)
    if dist.size == 0:
        return node_q, float("nan")
    order = np.argsort(nodes_idx, kind="stable")
    sorted_nodes = nodes_idx[order]
    sorted_dist = dist[order]
    present, starts = np.unique(sorted_nodes, return_index=True)
    stops = np.append(starts[1:], sorted_nodes.size)
    for node, a, b in zip(present, starts, stops):
        node_q[node] = np.percentile(sorted_dist[a:b], q_node)
    return node_q, float(np.percentile(dist, q_all))


@dataclass
class SOMModel:
    """A trained SOM frozen for inference.

    Attributes:
        xdim (int): x dimension of the lattice.
        ydim (int): y dimension of the lattice.
        weights (np.ndarray): float64 [xdim * ydim, F] node weights, node index = x + y * xdim.
        node_cluster (np.ndarray): int32 [nodes] cluster id of each node (-1 = unassigned).
        feature_names (list[str]): Name of each feature, in column order.
        scaler (Scaler): Scaling applied to raw data before the BMU search.
        node_dist_q99 (np.ndarray): float64 [nodes] 99th percentile of the Euclidean BMU
            distance of the training points mapped to each node (NaN if none).
        reject_distance (float): 99.9th percentile of the BMU distance over all training points.
        metadata (dict[str, str]): Provenance (format_version, awesom_version, created,
            git_commit, and user keys such as dataset, merge_cost, seed).
    """

    xdim: int
    ydim: int
    weights: np.ndarray
    node_cluster: np.ndarray
    feature_names: list[str]
    scaler: Scaler
    node_dist_q99: np.ndarray
    reject_distance: float
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.xdim = int(self.xdim)
        self.ydim = int(self.ydim)
        self.weights = np.ascontiguousarray(self.weights, dtype=np.float64)
        self.node_cluster = np.ascontiguousarray(self.node_cluster, dtype=np.int32).ravel()
        self.node_dist_q99 = np.ascontiguousarray(self.node_dist_q99, dtype=np.float64).ravel()
        self.feature_names = [str(n) for n in self.feature_names]
        self.reject_distance = float(self.reject_distance)
        n_nodes = self.xdim * self.ydim
        if self.weights.ndim != 2 or self.weights.shape[0] != n_nodes:
            raise ValueError(f"weights must have shape ({n_nodes}, F)")
        n_features = self.weights.shape[1]
        if self.node_cluster.shape != (n_nodes,) or self.node_dist_q99.shape != (n_nodes,):
            raise ValueError("node_cluster and node_dist_q99 need one entry per node")
        if len(self.feature_names) != n_features:
            raise ValueError("feature_names must have one entry per feature")
        for name in self.feature_names:
            if not name or "," in name or any(c.isspace() for c in name):
                raise ValueError(f"feature name {name!r} must be non-empty, without commas or whitespace")
        if self.scaler.offset.shape != (n_features,):
            raise ValueError("scaler must have one offset and scale per feature")
        self.metadata = _check_metadata(self.metadata)

    @property
    def n_nodes(self) -> int:
        """int: Number of lattice nodes."""
        return self.xdim * self.ydim

    @property
    def n_features(self) -> int:
        """int: Number of features."""
        return self.weights.shape[1]

    @classmethod
    def from_lattice(
        cls,
        lattice: "Lattice",
        scaled_training_data: np.ndarray | None = None,
        scaler: Scaler | None = None,
        merge_cost: float | None = None,
        metadata: dict | None = None,
    ) -> "SOMModel":
        """Freeze a trained ``Lattice``.

        If the lattice has no cluster assignment yet, ``assign_cluster_to_lattice``
        is called with ``merge_cost`` (0.0 if None) and its console output is
        suppressed.

        Args:
            lattice (Lattice): A trained lattice.
            scaled_training_data (np.ndarray, optional): [N, F] training data in the
                scaled space the lattice was trained in, used for the distance
                thresholds. Defaults to ``lattice.data_array``.
            scaler (Scaler, optional): Scaler that maps raw data to that space.
                Defaults to the identity.
            merge_cost (float, optional): Merge threshold for the cluster assignment;
                also recorded in the metadata. Defaults to None.
            metadata (dict, optional): Extra provenance; values are converted to str
                and override the defaults. Defaults to None.

        Returns:
            SOMModel: the frozen model.
        """
        weights = np.asarray(lattice.lattice, dtype=np.float64)
        xdim, ydim = int(lattice.xdim), int(lattice.ydim)
        n_features = weights.shape[1]

        clusters = getattr(lattice, "lattice_assigned_clusters", None)
        if clusters is None:
            with contextlib.redirect_stdout(io.StringIO()):
                clusters = lattice.assign_cluster_to_lattice(
                    merge_cost=0.0 if merge_cost is None else merge_cost
                )
        clusters = np.asarray(clusters)
        if clusters.shape != (xdim, ydim):
            raise ValueError(f"lattice_assigned_clusters must have shape ({xdim}, {ydim})")
        # clusters is indexed [x, y]; node index = x + y * xdim, i.e. Fortran order
        node_cluster = clusters.ravel(order="F").astype(np.int32)

        names = getattr(lattice, "features_names", None)
        if names is None:
            names = [f"f{i}" for i in range(n_features)]

        if scaled_training_data is None:
            scaled_training_data = lattice.data_array
        nodes_idx, dist2 = best_match_nodes(weights, scaled_training_data)
        node_q, reject = distance_quantiles(nodes_idx, np.sqrt(dist2), xdim * ydim)

        meta = _default_metadata()
        for key in ("seed", "train", "alpha_0"):
            if getattr(lattice, key, None) is not None:
                meta[key] = str(getattr(lattice, key))
        if merge_cost is not None:
            meta["merge_cost"] = str(merge_cost)
        meta["n_training_points"] = str(len(dist2))
        if metadata:
            meta.update({str(k): str(v) for k, v in metadata.items()})

        return cls(
            xdim=xdim,
            ydim=ydim,
            weights=weights.copy(),
            node_cluster=node_cluster,
            feature_names=list(names),
            scaler=scaler if scaler is not None else Scaler.identity(n_features),
            node_dist_q99=node_q,
            reject_distance=reject,
            metadata=meta,
        )

    def predict(
        self, data: np.ndarray, chunk: int = DEFAULT_CHUNK, return_distance: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Classify raw (unscaled) data.

        The scaler and the BMU search are applied ``chunk`` rows at a time, so the
        working memory is O(chunk) beyond the outputs.

        Args:
            data (np.ndarray): [N, F] float64 or float32 raw data.
            chunk (int, optional): Points per chunk. Defaults to 2**20.
            return_distance (bool, optional): Also return the Euclidean BMU distance
                in the scaled space. Defaults to False.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: int32 [N] labels, and float64
                [N] distances if ``return_distance`` is True.
        """
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] != self.n_features:
            raise ValueError(f"data must have shape (N, {self.n_features})")
        n_points = data.shape[0]
        labels = np.empty(n_points, dtype=np.int32)
        dist = np.empty(n_points, dtype=np.float64) if return_distance else None
        for start in range(0, n_points, chunk):
            stop = min(start + chunk, n_points)
            scaled = self.scaler.transform(data[start:stop])
            out = _predict(
                self.weights, self.node_cluster, scaled, chunk=chunk, return_distance=return_distance
            )
            if return_distance:
                labels[start:stop], dist[start:stop] = out
            else:
                labels[start:stop] = out
        return (labels, dist) if return_distance else labels

    # ------------------------------------------------------------------ I/O

    def save(self, path: str | Path) -> None:
        """Write the model to HDF5.

        Args:
            path (str | Path): Output file, conventionally ``*.h5``.
        """
        import h5py

        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = int(FORMAT_VERSION)
            f.attrs["xdim"] = self.xdim
            f.attrs["ydim"] = self.ydim
            f.attrs["reject_distance"] = self.reject_distance
            f.attrs["scaler_kind"] = self.scaler.kind
            f.attrs["bulk_range"] = self.scaler.bulk_range
            f.create_dataset("weights", data=self.weights)
            f.create_dataset("node_cluster", data=self.node_cluster)
            f.create_dataset("node_dist_q99", data=self.node_dist_q99)
            f.create_dataset(
                "feature_names",
                data=np.array(self.feature_names, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            g = f.create_group("scaler")
            g.create_dataset("offset", data=self.scaler.offset)
            g.create_dataset("scale", data=self.scaler.scale)
            m = f.create_group("metadata")
            for key, value in self.metadata.items():
                m.attrs[key] = value

    def save_text(self, path: str | Path) -> None:
        """Write the model as plain text (see the module docstring for the format).

        Args:
            path (str | Path): Output file, conventionally ``*.txt``.
        """
        lines = [
            f"# format {_TEXT_MAGIC}",
            f"# xdim {self.xdim}",
            f"# ydim {self.ydim}",
            f"# n_features {self.n_features}",
            f"# reject_distance {_fmt(self.reject_distance)}",
            f"# scaler_kind {self.scaler.kind}",
            f"# bulk_range {_fmt(self.scaler.bulk_range)}",
        ]
        lines += [f"# {k} {v}" for k, v in self.metadata.items()]
        lines.append("# feature_names " + ",".join(self.feature_names))
        lines.append("# scaler_offset " + " ".join(map(_fmt, self.scaler.offset)))
        lines.append("# scaler_scale " + " ".join(map(_fmt, self.scaler.scale)))
        lines.append("# node_dist_q99 " + " ".join(map(_fmt, self.node_dist_q99)))
        for w, c in zip(self.weights, self.node_cluster):
            lines.append(" ".join(map(_fmt, w)) + f" {int(c)}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SOMModel":
        """Read a model written by ``save`` (``.h5``/``.hdf5``) or ``save_text`` (anything else).

        Args:
            path (str | Path): Model file.

        Returns:
            SOMModel: the loaded model.
        """
        if Path(path).suffix.lower() in (".h5", ".hdf5"):
            return cls._load_hdf5(path)
        return cls._load_text(path)

    @classmethod
    def _load_hdf5(cls, path: str | Path) -> "SOMModel":
        import h5py

        with h5py.File(path, "r") as f:
            names = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in f["feature_names"][()]
            ]
            scaler = Scaler(
                kind=str(f.attrs["scaler_kind"]),
                offset=f["scaler/offset"][()],
                scale=f["scaler/scale"][()],
                bulk_range=float(f.attrs["bulk_range"]),
            )
            return cls(
                xdim=int(f.attrs["xdim"]),
                ydim=int(f.attrs["ydim"]),
                weights=f["weights"][()],
                node_cluster=f["node_cluster"][()],
                feature_names=names,
                scaler=scaler,
                node_dist_q99=f["node_dist_q99"][()],
                reject_distance=float(f.attrs["reject_distance"]),
                metadata={k: str(v) for k, v in f["metadata"].attrs.items()},
            )

    @classmethod
    def _load_text(cls, path: str | Path) -> "SOMModel":
        header: dict[str, str] = {}
        metadata: dict[str, str] = {}
        rows: list[list[str]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n").rstrip("\r")
                if not line.strip():
                    continue
                if line.startswith("#"):
                    key, _, value = line[1:].strip().partition(" ")
                    if key in _RESERVED_KEYS:
                        header[key] = value.strip()
                    else:
                        metadata[key] = value.strip()
                else:
                    rows.append(line.split())
        if header.get("format") != _TEXT_MAGIC:
            raise ValueError(f"{path} is not an aweSOM text model")

        xdim, ydim = int(header["xdim"]), int(header["ydim"])
        n_features = int(header["n_features"])
        if len(rows) != xdim * ydim or any(len(r) != n_features + 1 for r in rows):
            raise ValueError(f"{path}: expected {xdim * ydim} rows of {n_features} weights + cluster")

        def floats(key: str) -> np.ndarray:
            return np.array([float(v) for v in header[key].split()], dtype=np.float64)

        scaler = Scaler(
            kind=header["scaler_kind"],
            offset=floats("scaler_offset"),
            scale=floats("scaler_scale"),
            bulk_range=float(header["bulk_range"]),
        )
        return cls(
            xdim=xdim,
            ydim=ydim,
            weights=np.array([[float(v) for v in r[:-1]] for r in rows], dtype=np.float64),
            node_cluster=np.array([int(r[-1]) for r in rows], dtype=np.int32),
            feature_names=header["feature_names"].split(","),
            scaler=scaler,
            node_dist_q99=floats("node_dist_q99"),
            reject_distance=float(header["reject_distance"]),
            metadata=metadata,
        )
