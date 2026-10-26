"""B5: seed sweep throughput. K independent Lattice.train_lattice runs (different seeds) on the
real fiducial data, sequential vs a ThreadPoolExecutor of K threads. Needs the step kernel
compiled with nogil=True. Checks every threaded lattice is bitwise equal to its sequential twin.
usage: python bench_seed_threads.py [K=8] [train=100000] [reps=3]
"""
import json, sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(__file__))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from _common import load_fiducial, median_interleaved
import aweSOM
from aweSOM import Lattice

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
train = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
fid = load_fiducial()
data = np.ascontiguousarray(fid.data_array, dtype=np.float64)
names = list(fid.features_names)

def one(seed):
    som = Lattice(fid.xdim, fid.ydim, fid.alpha_0, train, sampling_type=fid.init)
    som.seed = seed
    som.train_lattice(data, names, history=False)
    return som.lattice

def seq():
    with contextlib.redirect_stdout(io.StringIO()):
        return [one(s) for s in range(K)]

def thr():
    with contextlib.redirect_stdout(io.StringIO()), ThreadPoolExecutor(K) as ex:
        return list(ex.map(one, range(K)))

t = median_interleaved({"sequential": seq, "threads": thr}, reps=reps, warm=1)
med = {k: float(np.median(v)) for k, v in t.items()}
a, b = seq(), thr()
print(json.dumps(dict(K=K, train=train, sequential_s=med["sequential"], threads_s=med["threads"],
    ratio=med["sequential"] / med["threads"], bitwise_equal=all(np.array_equal(x, y) for x, y in zip(a, b)),
    times=t, aweSOM=aweSOM.__file__)))
