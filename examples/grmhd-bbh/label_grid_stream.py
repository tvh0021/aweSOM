"""Stream exact patch BMUs once for several merge maps of the same lattice.

Each --output corresponds to one --model. Optional --level leaves other rows
as -1. Outputs are int16 NPY files and existing files are never replaced.
--backend auto uses cuda_screen when CuPy sees a CUDA device, else patch. Both
give the same nodes (lowest index on ties), so the labels do not depend on it.
"""
import argparse
import json
import resource
import time
from pathlib import Path
import h5py
import numpy as np
from aweSOM.model import SOMModel
from aweSOM.stream import BMUSearcher, cuda_available


def resolve_backend(choice):
    """'patch' or 'cuda_screen' for --backend auto/patch/cuda_screen."""
    if choice == 'patch': return 'patch'
    if cuda_available(): return 'cuda_screen'
    if choice == 'auto': return 'patch'
    raise RuntimeError('--backend cuda_screen: no usable CUDA device (CuPy and a visible GPU)')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--grid',required=True);p.add_argument('--model',action='append',required=True)
    p.add_argument('--output',action='append',required=True);p.add_argument('--level',type=int)
    p.add_argument('--backend',choices=['auto','patch','cuda_screen'],default='auto')
    args=p.parse_args()
    if len(args.model)!=len(args.output):raise ValueError('one output per model required')
    models=[SOMModel.load(path) for path in args.model]; ref=models[0]
    for m in models:
        if m.feature_names != ref.feature_names:
            raise ValueError("models must have identical feature names and order")
        if (m.xdim, m.ydim) != (ref.xdim, ref.ydim):
            raise ValueError("models must have identical lattice dimensions")
        assert np.array_equal(m.weights,ref.weights)
        assert np.array_equal(m.scaler.offset,ref.scaler.offset)
        assert np.array_equal(m.scaler.scale,ref.scaler.scale)
    for path in args.output:
        if Path(path).exists():raise FileExistsError(path)
    backend=resolve_backend(args.backend)
    search=BMUSearcher(ref.weights,backend,lattice_shape=(ref.xdim,ref.ydim))
    t=time.time()
    with h5py.File(args.grid) as h:
        names=[n.decode() if isinstance(n,bytes) else str(n) for n in h['names'][()]]
        if names != list(ref.feature_names):
            raise ValueError("grid and model feature names disagree")
        n=len(h['level']); outs=[np.lib.format.open_memmap(path,mode='w+',dtype='int16',shape=(n,)) for path in args.output]
        for a in range(0,n,1000000):
            sl=slice(a,min(a+1000000,n));lev=h['level'][sl]
            use=np.ones(len(lev),bool) if args.level is None else lev==args.level
            for o in outs:o[sl]=-1
            if use.any():
                x=h['features'][sl][use]
                if not np.isfinite(x).all():raise ValueError('nonfinite features')
                idx,_=search(ref.scaler.transform(x))
                for o,m in zip(outs,models):
                    lab=np.full(len(lev),-1,np.int16);lab[use]=np.asarray(m.node_cluster).ravel()[idx];o[sl]=lab
            if a%10000000==0:print(a+len(lev),'seconds',time.time()-t,flush=True)
        for o in outs:o.flush()
    print(json.dumps({'backend':backend,'seconds':time.time()-t,'peak_GB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9}),flush=True)

if __name__=='__main__':main()
