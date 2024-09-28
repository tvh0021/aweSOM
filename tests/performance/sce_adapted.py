## Original sce.py script from https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning.git
## This script is used to compare the performance of the aweSOM package against the original SCE implementation
## Small modifications where needed to make the script work with the aweSOM package output

import numpy as np

# import h5py as h5
import collections
from collections import defaultdict
import sys, os

# visualization
import matplotlib.pyplot as plt

from aweSOM.make_sce_clusters import make_file_name
from aweSOM.sce import load_som_npy
import glob


# create a normal dens enumpy array by first copying input array,
# then masking all elements not equal to cid away
# then setting those elements to 1, others to 0
def create_mask(img, cid):
    img_masked = np.copy(img)

    # create masked array
    mask_arr = np.ma.masked_where(img == cid, img)

    # set values mask to 1, 0 elsewhere
    img_masked[mask_arr.mask] = 1.0
    img_masked[~mask_arr.mask] = 0.0

    return img_masked


if __name__ == "__main__":

    # --------------------------------------------------
    # plotting env

    plt.rc("font", family="sans-serif")
    # plt.rc('text',  usetex=True)
    plt.rc("xtick", labelsize=5)
    plt.rc("ytick", labelsize=5)
    plt.rc("axes", labelsize=5)

    fig = plt.figure(1, figsize=(6, 6), dpi=300)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    gs = plt.GridSpec(1, 1)
    gs.update(hspace=0.05)
    gs.update(wspace=0.05)

    axs = []
    axs.append(plt.subplot(gs[0, 0]))

    # --------------------------------------------------
    # data

    # location of output directories of puhti1/2/3 runs

    runid = 0  # pre-computed sample runs id: 1/2/3
    if runid == 0:
        datadir = "som_out/"
    else:
        print("run not implemented yet")

    # list of runs
    runs = glob.glob(datadir + "/*.npy")

    # dictionary that stores mappings of runs and indices to count their score
    # if element is not here it automatically appends empty dict
    # mapping = collections.defaultdict(dict)
    # mapping = defaultdict(lambda : defaultdict(dict))

    class InfNestedDict(dict):
        """Implementation of perl's autovivification feature."""

        def __getitem__(self, item):
            try:
                return dict.__getitem__(self, item)
            except KeyError:
                value = self[item] = type(self)()
                return value

    mapping = InfNestedDict()

    multimap_mapping = InfNestedDict()

    # loop over data files reading image by image
    for run in runs:
        print("-----------------------")
        print(run)

        # clusters = read_clusterID(datadir, run)
        clusters = load_som_npy(run)
        nx = np.shape(clusters)

        # unique ids
        ids = np.unique(clusters)
        print("ids:", ids)
        nids = len(ids)  # number of cluster ids in this run

        # pick one/first image as a reference for plotting
        img = clusters

        # --------------------------------------------------
        if True:  # TRUE
            # map 2 map comparison
            #
            # build a raw mapping of ids by naively maximizing overlapping area
            # this is appended to `mappings` dictionary

            # loop over run1 ids
            for cid in range(nids):

                # create masked array where only id == cid are visible
                mask = create_mask(img, cid)

                total_mask = np.zeros(
                    np.shape(mask)
                )  # NOTE: total mask for base map (run) with cluster cid over all other maps (runC) and their clusters
                total_SQ_scalar = 0.0
                total_S_scalar = 0.0
                total_U_scalar = 0.0

                # mask_area_cid = np.ma.count_masked(mask)/(nx)
                mask_area_cid = np.sum(mask) / (nx)

                # loop over all other runs again to make run1 vs run2 comparison
                for runC in runs:

                    # dont compare to self
                    if run is runC:
                        continue

                    # print("    -----------------------")
                    # print("    ", runC)

                    # clustersC = read_clusterID(datadir, runC)
                    clustersC = load_som_npy(runC)
                    idsC = np.unique(clustersC)
                    imgC = clustersC

                    # loop over all ids in run2
                    for cidC in range(len(idsC)):
                        maskC = create_mask(imgC, cidC)
                        maskC_area_cidC = np.sum(maskC) / (nx)

                        # --------------------------------------------------
                        # product of two masked arrays; corresponds to intersection
                        I = mask * maskC

                        # count True values of merged array divided by total number of values
                        I_area = np.sum(I) / (nx)

                        # print('{}: {} vs {}: {} ='.format(
                        #    run, cid,
                        #    runC, cidC,
                        #    intersect_area))

                        # --------------------------------------------------
                        # sum of two masked arrays; corresponds to union
                        U = np.ceil((mask + maskC) * 0.5)  # ceil to make this 0s and 1s

                        # count True values of merged array divided by total number of values
                        U_area = np.sum(U) / (nx)

                        # --------------------------------------------------

                        # Intersection signal strength of two masked arrays, S
                        S = np.sum(I) / np.sum(U)
                        # S = I_area/U_area
                        S_matrix = S * I

                        # --------------------------------------------------
                        # Union quality of two masked arrays, Q
                        if np.sum(mask) == 0.0 or np.sum(maskC) == 0.0:
                            continue

                        # Q = U_area/(np.sum(mask)+np.sum(maskC))-I_area/(np.sum(mask)+np.sum(maskC))
                        Q = np.sum(U) / (np.sum(mask) + np.sum(maskC)) - np.sum(I) / (
                            np.sum(mask) + np.sum(maskC)
                        )
                        if Q == 0.0:
                            continue  # break here because this causes NaNs that accumulate. Why we get division by zero?

                        Q_matrix = Q * U

                        # --------------------------------------------------
                        # final measure for this comparison is (S/Q) x Union
                        SQ = S / Q

                        # normalize SQ with total map size to get smaller numbers (makes numerics easier)
                        # SQ /= nx

                        # SQ_matrix = SQ*I #NOTE: this is actually (S/Q)xI (not union)
                        # SQ_matrix = (SQ*U)
                        SQ_matrix = SQ * mask

                        # append these measures to the mapping dictionary
                        mapping[run][cid][runC][cidC] = (
                            total_S_scalar,
                            total_U_scalar,
                            SQ,
                            S,
                            Q,
                            U_area,
                            I_area,
                            mask_area_cid,
                            maskC_area_cidC,
                        )

                        total_mask += SQ_matrix  # pixelwise stacking of 2 masks
                        total_SQ_scalar += SQ
                        total_S_scalar += S
                        total_S_scalar += U
                        # print("    S/Q", SQ)
                        # print("    S", S)

                    # end of loop over runC cids
                # end of loop over runCs

                # --------------------------------------------------
                # total measure of this cluster id in this map is sum( S/Q )
                # total_SQ = np.sum(total_mask)/(nx)

                # skip self to self comparison
                if total_SQ_scalar == 0.0:
                    continue

                total_SQ_from_matrix = np.sum(total_mask) / (nx)

                multimap_mapping[run][cid] = (total_SQ_scalar, total_mask)

    # --------------------------------------------#
    # end of loop over runs

    # print multimap stacked values
    if True:  # TRUE
        print("multimap mappings:-----------------------")

        for map1 in mapping:
            print(map1)
            for id1 in mapping[map1]:
                # SQ, SQ_matrix = multimap_mapping[map1][id1]
                total_SQ_scalar, SQ_matrix = multimap_mapping[map1][id1]
                # print("   ", id1, total_SQ_scalar)
