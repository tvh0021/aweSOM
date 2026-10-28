import sys

import h5py

for p in sys.argv[1:]:
    f = h5py.File(p, "r")

    def show(n, o):
        if isinstance(o, h5py.Dataset):
            print(" ", n, o.shape, o.dtype, o.chunks, o.compression)

    print(p)
    f.visititems(show)
    for k, v in f.attrs.items():
        print("   attr", k, repr(v)[:200])
