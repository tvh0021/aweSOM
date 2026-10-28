"""Measure full-grid sheet fractions and slab EDT with disk-backed masks.

Run after chunked prediction. No result file is replaced. Masks and statistics
are written into a new output directory. Spacings follow the BBH level convention.
"""
import argparse
import json
import resource
import time
from pathlib import Path
import h5py
import numpy as np
from aweSOM.sheet_stats import LevelLattice
from aweSOM.slab_thickness import ridge_thickness


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--grid',required=True); p.add_argument('--labels',required=True)
    p.add_argument('--out',required=True); p.add_argument('--clusters',type=int,nargs='+',required=True)
    args=p.parse_args(); out=Path(args.out); out.mkdir(exist_ok=False)
    labels=np.load(args.labels,mmap_mode='r'); chunk=1000000; t=time.time()
    with h5py.File(args.grid) as h:
        ranges={}; bounds={}
        for a in range(0,len(labels),chunk):
            lev=h['level'][a:a+chunk]; xyz=h['xyz'][a:a+chunk]
            for L in np.unique(lev):
                L=int(L); ii=np.flatnonzero(lev==L); lo=a+int(ii[0]); hi=a+int(ii[-1])+1
                if L not in ranges: ranges[L]=[lo,hi]; bounds[L]=[xyz[ii].min(0),xyz[ii].max(0)]
                else:
                    assert ranges[L][1]==lo
                    ranges[L][1]=hi
                    bounds[L]=[np.minimum(bounds[L][0],xyz[ii].min(0)),np.maximum(bounds[L][1],xyz[ii].max(0))]
                assert hi-lo==len(ii)
        lats={}; owned={}; sheets={}; result={'levels':{},'between':{}}
        holes=np.asarray(h.attrs['horizon_centres']).reshape(2,3)
        seg=holes[1]-holes[0]; length=np.linalg.norm(seg); u=seg/length
        v=np.cross([0,0,1],u)
        if np.linalg.norm(v)<1e-12:v=np.array([1.,0.,0.])
        v/=np.linalg.norm(v); wv=np.cross(u,v)
        for L,(a,b) in ranges.items():
            dx=2.**(3-L); lo,hi=bounds[L]; shape=tuple(map(int,np.rint((hi-lo)/dx).astype(int)+1))
            lat=lats[L]=LevelLattice(L,dx,tuple(map(float,lo)),shape)
            own=np.lib.format.open_memmap(out/f'owned_{L}.npy',mode='w+',dtype=bool,shape=shape)
            sh=np.lib.format.open_memmap(out/f'sheet_{L}.npy',mode='w+',dtype=bool,shape=shape)
            n_sheet=0; bt={w:[0,0] for w in (2,4,8)}
            for start in range(a,b,chunk):
                stop=min(start+chunk,b); pos=h['xyz'][start:stop]; ins=np.isin(labels[start:stop],args.clusters)
                idx=lat.flat(pos); own.ravel()[idx]=True; sh.ravel()[idx[ins]]=True; n_sheet+=int(ins.sum())
                rel=pos.astype(float)-holes[0]; s=rel@u/length; pa=np.abs(rel@v); pb=np.abs(rel@wv)
                for w in bt:
                    reg=(s>=0)&(s<=1)&(pa<=w)&(pb<=w)
                    bt[w][0]+=int(reg.sum());bt[w][1]+=int((reg&ins).sum())
            own.flush();sh.flush();del own,sh
            owned[L]=np.load(out/f'owned_{L}.npy',mmap_mode='r');sheets[L]=np.load(out/f'sheet_{L}.npy',mmap_mode='r')
            result['levels'][L]={'n':b-a,'sheet_n':n_sheet,'fraction':n_sheet/(b-a),'dx':dx,'shape':shape,
                                  'owned_M3':(b-a)*dx**3,'sheet_M3':n_sheet*dx**3,'between_counts':bt}
            print('mask',L,result['levels'][L],flush=True)
        total=sum(r['owned_M3'] for r in result['levels'].values()); sheet=sum(r['sheet_M3'] for r in result['levels'].values())
        result.update(owned_M3=total,sheet_M3=sheet,fraction=sheet/total)
        for w in (2,4,8):
            tv=sum(r['between_counts'][w][0]*r['dx']**3 for r in result['levels'].values())
            sv=sum(r['between_counts'][w][1]*r['dx']**3 for r in result['levels'].values())
            result['between'][w]={'owned_M3':tv,'sheet_M3':sv,'filling':sv/tv,'share':sv/sheet,'enhancement':sv/tv/(sheet/total)}
        for L in lats:
            print('EDT start',L,lats[L],flush=True)
            thick=ridge_thickness(L,lats,sheets,owned)
            result['levels'][L]['thickness']=thick
            print('EDT done',L,thick,flush=True)
        result['seconds']=time.time()-t;result['peak_GB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9
        with open(out/'statistics.json','x') as f:json.dump(result,f,indent=2)

if __name__=='__main__':main()
