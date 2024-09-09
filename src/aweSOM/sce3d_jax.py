## Statistically Combined Ensemble (SCE) code for 3D data, originally described in Bussov & Nattila (2021)
## JAX version, GPU and JIT accelerated. Capable of handling 1000^3 data with 1 A100-80GB GPU. 

import numpy as np
import os
import glob
import argparse

# import jax for GPU computation
import jax.numpy as jnp
from jax import jit

# read array from clusterID.npy
def load_som_npy(path : str) -> jnp.ndarray:
    """Load a numpy array from a file onto the GPU"""
    return jnp.load(path, 'r')

@jit
def create_mask(img : jnp.ndarray, cid : int) -> jnp.ndarray:
    """Create a mask for a given cluster id

    Args:
        img (jnp.ndarray): 3D array of cluster ids
        cid (int): cluster id to mask

    Returns:
        jnp.ndarray: masked cluster, 1 where cluster id is cid, 0 elsewhere
    """
    return jnp.where(img == cid, 1, 0)

def compute_SQ(mask : jnp.ndarray, maskC : jnp.ndarray):
    """Compute the quality index between two masks

    Args:
        mask (jnp.ndarray): _description_
        maskC (jnp.ndarray): _description_

    Returns:
        SQ (float): quality index, equals to S/Q
        SQ_matrix (jnp.ndarray): pixelwise quality index, equals to S/Q * mask
    """
    #--------------------------------------------------
    # product of two masked arrays; corresponds to intersection
    I = jnp.multiply(mask, maskC)

    #--------------------------------------------------
    # sum of two masked arrays; corresponds to union
    U = jnp.ceil((mask + maskC) * 0.5)
    # U_area = jnp.sum(U) / (nx * ny * nz)
    
    #--------------------------------------------------
    # Intersection signal strength of two masked arrays, S
    S = jnp.sum(I) / jnp.sum(U)

    #--------------------------------------------------
    # Union quality of two masked arrays, Q
    if jnp.max(mask) == 0  or jnp.max(maskC) == 0:
        return 0., jnp.zeros(mask.shape), 0., 0.
    
    Q = jnp.sum(U) / (jnp.sum(mask) + jnp.sum(maskC)) - jnp.sum(I) / (jnp.sum(mask) + jnp.sum(maskC))
    if Q == 0.0:
        return 0., jnp.zeros(mask.shape) #break here because this causes NaNs that accumulate.
    
    #--------------------------------------------------
    # final measure for this comparison is (S/Q) x Union
    SQ = S / Q
    SQ_matrix = SQ * mask

    return SQ, SQ_matrix
    

def nested_loop(all_data, number_of_clusters):
    runs = all_data
    
    # loop over data files reading image by image
    for i in range(len(runs)):
        run = runs[i]
        
        clusters_1d = load_som_npy(run)
        print('-----------------------')
        print("Run : ", run, flush=True)

        with open(subfolder + '/multimap_mappings.txt', 'a') as f:
            f.write('{}\n'.format(run))
        
        # nx x ny x nz size maps
        nz,ny,nx = jnp.cbrt(clusters_1d.shape[0]).astype(int), jnp.cbrt(clusters_1d.shape[0]).astype(int), jnp.cbrt(clusters_1d.shape[0]).astype(int)
        clusters = clusters_1d.reshape(nz,ny,nx)

        # unique ids
        nids = number_of_clusters[i] #number of cluster ids in this run
        # ids = np.arange(nids)
        print('nids : ', nids)
        
        
        for cid in range(nids):
            print('  -----------------------')
            print('  cid : ', cid, flush=True)

            # create masked array where only id == cid are visible
            mask = create_mask(clusters, cid)

            total_mask = jnp.zeros((ny,nx,nz), dtype=float)
            
            total_SQ_scalar = 0.
            
            for j in range(len(runs)):
                runC = runs[j]
                
                if j == i: # don't compare to itself
                    continue
                
                clustersC_1d = load_som_npy(runC)
                clustersC = clustersC_1d.reshape(nz,ny,nx)

                print('    -----------------------')
                print('   ',runC, flush=True)

                nidsC = number_of_clusters[j] #number of cluster ids in this run
                print('    nidsC : ', nidsC)

                for cidC in range(nidsC):
                    maskC = create_mask(clustersC, cidC)
                    
                    SQ, SQ_matrix = compute_SQ(mask, maskC)
                    
                    # pixelwise stacking of 2 masks
                    total_mask += SQ_matrix # for numpy array
                    
                    total_SQ_scalar += SQ
            
            # save total mask to file
            print("Saving total mask to file", flush=True)
            jnp.save(subfolder + '/mask3d-{}-id{}.npy'.format(run, cid), total_mask)

            # multimap_mapping[run][cid] = total_SQ_scalar
            print("Saving total SQ scalar to multimap_mapping", flush=True)
            with open(subfolder + '/multimap_mappings.txt', 'a') as f:
                f.write('{} {}\n'.format(cid, total_SQ_scalar))
        
    return 0
                    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='SCE code')
    parser.add_argument('--folder', type=str, dest='folder', default=os.getcwd(), help='Folder name')
    parser.add_argument('--subfolder', type=str, dest='subfolder', default='SCE', help='Subfolder name')

    args = parser.parse_args()

    print("Starting SCE3d", flush=True)
    folder = args.folder
    os.chdir(folder)
    cluster_files = glob.glob('*.npy')

    #--------------------------------------------------
    # data
    subfolder = args.subfolder
    print(cluster_files)
    
    #--------------------------------------------------
    # calculate unique number of clusters per run
    
    nids_array = np.empty(len(cluster_files), dtype=int)
    for run in range(len(cluster_files)):
        clusters = load_som_npy(cluster_files[run])
        ids = jnp.unique(clusters)
        nids_array[run] = len(ids)
        # print (nids_array[run])
    
    print('nids_array:', nids_array, flush=True)
    print("There are {} runs".format(len(cluster_files)), flush=True)
    print("There are {} clusters in total".format(np.sum(nids_array)), flush=True)

    #--------------------------------------------------
    # generate index for multimap_mapping as the loop runs. Avoid declaring a dict beforehand to avoid memory leaks
    with open (subfolder + '/multimap_mappings.txt', 'w') as f:
        f.write('')

    #--------------------------------------------------
    # loop over data files reading image by image and do pairwise comparisons
    # all wrapped inside the nested_loop function, which uses JAX for fast computation
    nested_loop(cluster_files, nids_array)