## Original sce.py script from https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning.git
## This script is used to compare the performance of the aweSOM package against the original SCE implementation

import numpy as np
import h5py as h5
import collections
from collections import defaultdict
import sys, os

# visualization
import matplotlib.pyplot as plt


# visualize matrix
def imshow(
    ax,
    grid,
    xmin,
    xmax,
    ymin,
    ymax,
    cmap="plasma",
    vmin=0.0,
    vmax=1.0,
    clip=-1.0,
    cap=None,
    aspect="auto",
    plot_log=False,
):

    ax.clear()
    ax.minorticks_on()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # ax.set_xlim(-3.0, 3.0)
    # ax.set_ylim(-3.0, 3.0)

    extent = [xmin, xmax, ymin, ymax]

    if clip == None:
        mgrid = grid
    elif type(clip) == tuple:
        cmin, cmax = clip
        print(cmin, cmax)
        mgrid = np.ma.masked_where(np.logical_and(cmin <= grid, grid <= cmax), grid)
    else:
        mgrid = np.ma.masked_where(grid <= clip, grid)

    if cap != None:
        mgrid = np.clip(mgrid, cap)

    if plot_log:
        mgrid = np.log10(mgrid)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)

    mgrid = mgrid.T
    im = ax.imshow(
        mgrid,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        # vmax = Nrank,
        # alpha=0.5
    )
    return im


# read array from clusterID.h5
def read_clusterID(datadir, run):
    datafile = (
        datadir + "/" + run + "/data_som_clusters_13000.h5"
    )  # NOTE: hardcoded data file name
    f5 = h5.File(datafile, "r")
    clusters = f5["databack"][()]
    f5.close()
    return clusters


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
        datadir = "sample-som-runs"
        runs = [
            "output_15_0.6_50000",  ##
            "output_15_0.7_50000",  ##
            "output_15_0.8_50000",  ##
        ]
    else:
        print("run not implemented yet")

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

        clusters = read_clusterID(datadir, run)

        # nx x ny size maps and ni subimages
        nx, ny, ni = np.shape(clusters)

        # image size
        xmin, xmax = 0, nx
        ymin, ymax = 0, ny

        # unique ids
        ids = np.unique(clusters[:, :, 0])
        print("ids:", ids)
        nids = len(ids)  # number of cluster ids in this run

        # visualize first image as an example
        if True:
            imshow(
                axs[0],
                clusters[:, :, 0],
                xmin,
                xmax,
                ymin,
                ymax,
                vmin=0.0,
                vmax=nids,
                cmap="Spectral",
            )
            fig.savefig("stack_{}.png".format(run))

        # pick one/first image as a reference for plotting
        img = clusters[:, :, 0]

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

                # mask_area_cid = np.ma.count_masked(mask)/(nx*ny)
                mask_area_cid = np.sum(mask) / (nx * ny)

                # loop over all other runs again to make run1 vs run2 comparison
                for runC in runs:

                    # dont compare to self
                    if run is runC:
                        continue

                    print("    -----------------------")
                    print("    ", runC)

                    clustersC = read_clusterID(datadir, runC)
                    idsC = np.unique(clustersC[:, :, 0])
                    imgC = clustersC[:, :, 0]

                    # loop over all ids in run2
                    for cidC in range(len(idsC)):
                        maskC = create_mask(imgC, cidC)
                        maskC_area_cidC = np.sum(maskC) / (nx * ny)

                        # --------------------------------------------------
                        # product of two masked arrays; corresponds to intersection
                        I = mask * maskC

                        # count True values of merged array divided by total number of values
                        I_area = np.sum(I) / (nx * ny)

                        # print('{}: {} vs {}: {} ='.format(
                        #    run, cid,
                        #    runC, cidC,
                        #    intersect_area))

                        if False:
                            # print('plotting intersect...')
                            imshow(
                                axs[0],
                                I,
                                xmin,
                                xmax,
                                ymin,
                                ymax,
                                vmin=0.0,
                                vmax=1.0,
                                cmap="binary",
                            )
                            fig.savefig(
                                datadir
                                + "/intersect_map1-{}_map2-{}_id1-{}_id2-{}.png".format(
                                    run, runC, cid, cidC
                                )
                            )

                        # --------------------------------------------------
                        # sum of two masked arrays; corresponds to union
                        U = np.ceil((mask + maskC) * 0.5)  # ceil to make this 0s and 1s

                        # count True values of merged array divided by total number of values
                        U_area = np.sum(U) / (nx * ny)

                        if False:
                            # print('plotting union...')
                            imshow(
                                axs[0],
                                U,
                                xmin,
                                xmax,
                                ymin,
                                ymax,
                                vmin=0.0,
                                vmax=1.0,
                                cmap="binary",
                            )
                            fig.savefig(
                                datadir
                                + "/union_map1-{}_map2-{}_id1-{}_id2-{}.png".format(
                                    run, runC, cid, cidC
                                )
                            )

                        # --------------------------------------------------

                        # Intersection signal strength of two masked arrays, S
                        S = np.sum(I) / np.sum(U)
                        # S = I_area/U_area
                        S_matrix = S * I

                        if False:
                            print("plotting intersect area...", S)
                            imshow(
                                axs[0],
                                S_matrix,
                                xmin,
                                xmax,
                                ymin,
                                ymax,
                                vmin=0.0,
                                vmax=S,
                                cmap="seismic",
                            )
                            fig.savefig(
                                datadir
                                + "/signalstrength_map1-{}_map2-{}_id1-{}_id2-{}.png".format(
                                    run, runC, cid, cidC
                                )
                            )

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

                        if False:
                            print("plotting quality...", Q)
                            imshow(
                                axs[0],
                                Q_matrix,
                                xmin,
                                xmax,
                                ymin,
                                ymax,
                                vmin=0.0,
                                vmax=Q,
                                cmap="YlGn",
                            )
                            fig.savefig(
                                datadir
                                + "/quality_map1-{}_map2-{}_id1-{}_id2-{}.png".format(
                                    run, runC, cid, cidC
                                )
                            )

                        # --------------------------------------------------
                        # final measure for this comparison is (S/Q) x Union
                        SQ = S / Q

                        # normalize SQ with total map size to get smaller numbers (makes numerics easier)
                        # SQ /= nx*ny

                        # SQ_matrix = SQ*I #NOTE: this is actually (S/Q)xI (not union)
                        # SQ_matrix = (SQ*U)
                        SQ_matrix = SQ * mask

                        if False:
                            print("plotting SQU...", SQ)
                            imshow(
                                axs[0],
                                SQ_matrix,
                                xmin,
                                xmax,
                                ymin,
                                ymax,
                                vmin=0.0,
                                vmax=SQ,
                                cmap="plasma",
                                plot_log=True,
                            )
                            fig.savefig(
                                datadir
                                + "/SQU_map1-{}_map2-{}_id1-{}_id2-{}.png".format(
                                    run, runC, cid, cidC
                                )
                            )

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

                        total_mask[:, :] += SQ_matrix[
                            :, :
                        ]  # pixelwise stacking of 2 masks
                        total_SQ_scalar += SQ
                        total_S_scalar += S
                        total_S_scalar += U
                        print("    S/Q", SQ)
                        print("    S", S)

                    # end of loop over runC cids
                # end of loop over runCs

                # --------------------------------------------------
                # total measure of this cluster id in this map is sum( S/Q )
                # total_SQ = np.sum(total_mask)/(nx*ny)

                # skip self to self comparison
                if total_SQ_scalar == 0.0:
                    continue

                total_SQ_from_matrix = np.sum(total_mask) / (nx * ny)

                multimap_mapping[run][cid] = (total_SQ_scalar, total_mask)

                if True:  # TRUE
                    # print('plotting total SQU...', np.log10(total_SQ), np.log10(np.min(total_mask)), np.log10(np.max(total_mask)) )
                    print(
                        "plotting total SQU:",
                        total_SQ_scalar,
                        "vs sum",
                        total_SQ_from_matrix,
                        " min:",
                        np.min(total_mask),
                        " max:",
                        np.max(total_mask),
                    )
                    imshow(
                        axs[0],
                        total_mask,
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                        vmin=0.0,  # np.min(total_mask),
                        vmax=np.max(
                            total_mask
                        ),  # 10, np.max(total_mask), #NOTE: 1e7 is about maximum value we seem to get
                        cmap="Reds",
                    )
                    fig.savefig(datadir + "/SQ_map1-{}_id1-{}.png".format(run, cid))

                    # log version
                    imshow(
                        axs[0],
                        total_mask,
                        xmin,
                        xmax,
                        ymin,
                        ymax,
                        vmin=np.min(total_mask),  # 0.1 np.min(total_mask),
                        vmax=np.max(
                            total_mask
                        ),  # 10, np.max(total_mask), #NOTE: 1e7 is about maximum value we seem to get
                        cmap="Blues",
                        plot_log=True,
                    )
                    fig.savefig(datadir + "/SQ_map1-{}_id1-{}_log.png".format(run, cid))

                    print("\n")

    # --------------------------------------------#
    # end of loop over runs

    # print all map2map comparison values
    if False:
        print("mappings:-----------------------")
        # print(mapping)

        for map1 in mapping:
            print(map1)
            for id1 in mapping[map1]:
                print(" ", id1)
                for map2 in mapping[map1][id1]:
                    print("  ", map2)

                    # best mapping of map1 id1 <-> map2 id2 is the one with the largest val
                    # this is one measure of how good the correspondence is

                    for id2 in mapping[map1][id1][map2]:
                        val = mapping[map1][id1][map2][id2]

                        # values we print here are:
                        # (intersect_area, union_area, area(map1, id1), area(map2,id2))
                        print("   ", id2, val)

    # print multimap stacked values
    if True:  # TRUE
        print("multimap mappings:-----------------------")

        for map1 in mapping:
            print(map1)
            for id1 in mapping[map1]:
                # SQ, SQ_matrix = multimap_mapping[map1][id1]
                total_SQ_scalar, SQ_matrix = multimap_mapping[map1][id1]
                print("   ", id1, total_SQ_scalar)
