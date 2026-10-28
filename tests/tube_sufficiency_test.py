"""Slab selection, selective I/O and cross-hole holdout contracts."""
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples/grmhd-bbh'))
from tube_sufficiency_hhr import slab_mask, selected_features, cross_hole_auc


def test_slab_is_level_relative_strict_and_box_limited():
    xyz=np.array([[0,0,.1],[0,0,.1],[0,0,.0625],[12.01,0,0],[0,0,-.03]])
    mask=slab_mask(xyz,np.array([5,8,8,8,8]),np.zeros((1,3)),np.zeros(3),np.array([0,0,1]))
    assert mask.tolist()==[True,False,False,False,True]


def test_selective_reader_reads_only_retained_contiguous_runs():
    class RecordingDataset:
        shape=(20,2)
        dtype=np.dtype('float32')
        calls=[]
        def __getitem__(self,key):
            self.calls.append((key.start,key.stop))
            return np.arange(40,dtype=self.dtype).reshape(20,2)[key]
    ds=RecordingDataset()
    rows=np.array([1,2,7,10,11,12])
    np.testing.assert_equal(selected_features(ds,rows),np.arange(40).reshape(20,2)[rows])
    assert ds.calls==[(1,3),(7,8),(10,13)]


def test_holdout_learns_sign_on_training_hole_without_test_reorientation():
    # Opposite relationship across holes must score below chance, not be flipped.
    X=np.array([[-2],[-1],[1],[2],[2],[1],[-1],[-2]],float)
    classes=np.tile([0,0,1,1],2)
    holes=np.repeat([1,2],4)
    folds=cross_hole_auc(X,classes,holes,0,[0])
    assert [f['auc'] for f in folds]==[0,0]
    assert folds[0]['train_hole']==1 and folds[1]['train_hole']==2
    missing=cross_hole_auc(X,classes,np.ones(8),0,[0])
    assert all(f['auc'] is None for f in missing)
