"""Profile the current Lattice.train_lattice for 100k steps on a 64x37 lattice."""
import cProfile, pstats, io, contextlib
import numpy as np
from aweSOM import Lattice

rng = np.random.default_rng(0)
data = rng.normal(size=(200_000, 4)) * 0.5
som = Lattice(64, 37, 0.1, 100_000)
som.train_lattice(data[:2000], ["a", "b", "c", "d"], number_of_steps=500)  # JIT warm-up
som = Lattice(64, 37, 0.1, 100_000)
pr = cProfile.Profile()
with contextlib.redirect_stdout(io.StringIO()):
    pr.enable(); som.train_lattice(data, ["a", "b", "c", "d"]); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8); print(s.getvalue()[:2500])
