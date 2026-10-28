"""Train a SOM on a large HDF5 feature file with the streaming CPU batch trainer, then label every row.

Driver for ``train_stream_lagoon.sbatch``; also runs on a laptop for small files.
Configuration comes from environment variables:

  AWESOM_FEATURES   HDF5 file with a 2-D (N, F) feature dataset (required)
  AWESOM_DATASET    dataset name (default "features"); an optional "names"
                    dataset with the feature names is copied to the model
  AWESOM_OUT        output directory (default ./awesom_stream_<jobid>)
  XDIM, YDIM        lattice (default 91 x 64)
  EPOCHS            batch epochs (default 40)
  MODE              "sample" (default): train on a stratified in-memory sample of
                    SAMPLE_ROWS rows, then label all N rows by streaming.
                    "full": train on all rows, streaming (or from RAM with
                    LOAD_TO_RAM=1), each epoch using SAMPLE_FRACTION of the rows.
  SAMPLE_ROWS       rows of the training sample in MODE=sample (default 1e8)
  SAMPLE_FRACTION   rows per epoch in MODE=full (default 1.0 = all)
  SAMPLE_BLOCK      rows per sampled block (default 4096; rounded up to a
                    multiple of the HDF5 chunk length so every block read is
                    whole chunks)
  LOAD_TO_RAM       1: read the whole dataset into memory first (MODE=full;
                    N x F x 4 bytes, 200 GB for 10^10 x 5)
  SCALE             "manual" (default; aweSOM manual scaling fitted on the
                    sample) or "none" (features are already scaled)
  LABELS            1 (default): write the BMU node of every row to
                    AWESOM_OUT/labels.h5 ("bmu", int16 when nodes < 32768)
  CHUNK             rows per streamed chunk (default 2**21)
  SEED              seed (default 42)

Outputs in AWESOM_OUT: model.h5 (weights, lattice, scaler, per-epoch
diagnostics), labels.h5, and a JSON log line per stage on stdout.
"""

from __future__ import annotations

import json
import os
import time

import h5py
import numba
import numpy as np

from aweSOM.model import Scaler
from aweSOM.stream import peak_rss_bytes, predict_stream, stratified_sample, train_batch_stream


def env(name, default, cast=str):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


def log(stage, **kw):
    rec = {"stage": stage, "time": time.strftime("%H:%M:%S"), "load": os.getloadavg(),
           "threads": numba.get_num_threads(), "peak_rss_gb": peak_rss_bytes() / 1e9, **kw}
    print(json.dumps(rec, default=float), flush=True)


class Scaled:
    """Row-sliceable view of an h5py dataset (or array) that applies a Scaler and returns float32."""

    def __init__(self, ds, scaler: Scaler):
        self.ds, self.scaler = ds, scaler
        self.shape = tuple(ds.shape)
        self.dtype = np.dtype(np.float32)

    def __getitem__(self, key):
        return self.scaler.transform(np.asarray(self.ds[key])).astype(np.float32)


def main() -> None:
    path = os.environ["AWESOM_FEATURES"]
    dataset = env("AWESOM_DATASET", "features")
    out_dir = env("AWESOM_OUT", f"awesom_stream_{os.environ.get('SLURM_JOB_ID', 'local')}")
    xdim, ydim = env("XDIM", 91, int), env("YDIM", 64, int)
    epochs = env("EPOCHS", 40, int)
    mode = env("MODE", "sample")
    sample_rows = int(env("SAMPLE_ROWS", 1e8, float))
    fraction = env("SAMPLE_FRACTION", 1.0, float)
    block = env("SAMPLE_BLOCK", 4096, int)
    load_to_ram = env("LOAD_TO_RAM", 0, int) == 1
    scale_kind = env("SCALE", "manual")
    write_labels = env("LABELS", 1, int) == 1
    chunk = env("CHUNK", 2**21, int)
    seed = env("SEED", 42, int)
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()

    with h5py.File(path, "r") as f5:
        ds = f5[dataset]
        n, n_feat = ds.shape
        names = [s.decode() if isinstance(s, bytes) else str(s) for s in f5["names"][()]] if "names" in f5 else [
            f"f{i}" for i in range(n_feat)]
        if ds.chunks is not None:
            c_rows = ds.chunks[0]
            block = max(c_rows, -(-block // c_rows) * c_rows)
        log("open", file=path, dataset=dataset, N=n, F=n_feat, hdf5_chunks=ds.chunks,
            compression=ds.compression, mode=mode, xdim=xdim, ydim=ydim, epochs=epochs, block=block)

        # stratified sample: training set (MODE=sample) and scaler fit
        t0 = time.perf_counter()
        n_fit = sample_rows if mode == "sample" else min(sample_rows, 10_000_000)
        raw_sample = stratified_sample(ds, n_fit, block=block, seed=seed, chunk=chunk)
        scaler = Scaler.fit(raw_sample, kind=scale_kind)
        log("sample", rows=len(raw_sample), seconds=time.perf_counter() - t0,
            offset=scaler.offset.tolist(), scale=scaler.scale.tolist())

        scaled_all = Scaled(ds, scaler)
        if mode == "sample":
            train_src = scaler.transform(raw_sample).astype(np.float32)
            del raw_sample
            kw = dict(final_pass="none")
        elif mode == "full":
            del raw_sample
            if load_to_ram:
                t0 = time.perf_counter()
                train_src = np.empty((n, n_feat), np.float32)
                for s in range(0, n, chunk):
                    train_src[s : s + chunk] = scaled_all[s : s + chunk]
                log("load_to_ram", seconds=time.perf_counter() - t0)
            else:
                train_src = scaled_all
            kw = dict(sample_fraction=fraction, sample_block=block, final_pass="none")
        else:
            raise SystemExit(f"MODE must be sample or full, got {mode}")

        t0 = time.perf_counter()
        weights, info = train_batch_stream(train_src, xdim, ydim, epochs=epochs, chunk=chunk,
                                           seed=seed, verbose=True, **kw)
        t_train = time.perf_counter() - t0
        log("train", seconds=t_train, rows_per_epoch=int(info["rows_per_epoch"][0]),
            epoch_time_mean=float(info["epoch_time"].mean()), qe_last_entering=float(info["qe"][-2]) if epochs > 1 else None,
            bmu=info["bmu_kind"][-1])
        # label from the in-RAM copy if there is one (MODE=full, LOAD_TO_RAM=1), else re-read the file
        label_src = train_src if (mode == "full" and load_to_ram) else scaled_all
        del train_src

        with h5py.File(os.path.join(out_dir, "model.h5"), "w") as m5:
            m5["weights"] = weights
            m5["feature_names"] = np.array(names, dtype="S")
            m5["scaler_offset"] = scaler.offset
            m5["scaler_scale"] = scaler.scale
            for k in ("radius", "qe", "epoch_time", "rows_per_epoch", "io_wait", "bmu_time"):
                m5[f"info/{k}"] = info[k]
            m5.attrs.update({"xdim": xdim, "ydim": ydim, "scaler_kind": scaler.kind, "mode": mode,
                             "epochs": epochs, "n_total": n, "source": path, "dataset": dataset,
                             "seed": seed, "node_index": "x + y * xdim"})

        if write_labels:
            t0 = time.perf_counter()
            nodes = xdim * ydim
            dtype = np.int16 if nodes < 32768 else np.int32
            with h5py.File(os.path.join(out_dir, "labels.h5"), "w") as l5:
                out = l5.create_dataset("bmu", shape=(n,), dtype=dtype, chunks=(min(n, 2**20),))
                out.attrs["meaning"] = "best-matching node index (x + y * xdim) of every row"
                buf_writer = _ChunkWriter(out)
                _, stats = predict_stream(weights, np.arange(nodes, dtype=np.int32), label_src,
                                          out=buf_writer, chunk=chunk, backend="patch",
                                          lattice_shape=(xdim, ydim))
            t_lab = time.perf_counter() - t0
            log("labels", wall_seconds=t_lab, mpts_s=n / t_lab / 1e6,
                **{f"predict_{k}": v for k, v in stats.items()})
    log("done", total_seconds=time.perf_counter() - t_start, out=out_dir)


class _ChunkWriter:
    """Accepts ``w[a:b] = labels`` and writes to an h5py dataset with the narrower dtype."""

    def __init__(self, ds):
        self.ds = ds

    def __setitem__(self, key, value):
        self.ds[key] = np.asarray(value).astype(self.ds.dtype)


if __name__ == "__main__":
    main()
