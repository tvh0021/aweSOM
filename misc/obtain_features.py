## This script is used to obtain the feature set for SOM training; which works 
## specifically for PIC simulation data, e.g. see Nattila & Beloborodov (2021)

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py as h5
import argparse
from scipy.ndimage import zoom, rotate, convolve

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth = 200)

# pytools bindings
import pytools
from initialize_turbulence import Configuration_Turbulence as Configuration

from multiprocessing import Pool
from multiprocessing import cpu_count

default_values = {
    "cmap": "hot",
    "vmin": None,
    "vmax": None,
    "clip": None,
    "aspect": 1,
    "vsymmetric": None,
    "winsorize_min": 0.005,
    "winsorize_max": 0.005,
    "title": "",
    "derived": False,
    "file": "flds",
    "log": False,
    "vmin": -1,
    "vmax": +1,
}


default_turbulence_values = {
    "rho": {
        "title": r"$n/n_0$",
        "vmin": 0.0,
        "vmax": 4.0,
    },
    "jz": {
        "title": r"$J_z$",
        "cmap": "RdBu",
        "vsymmetric": True,
        "vmin": -1.0000,
        "vmax": 1.0000,
    },
    "bz": {
        "title": r"$B_z$",
        "cmap": "RdBu",
        "vsymmetric": True,
    },
    "bperp": {
        "title": r"$B_\perp$",
        "cmap": "magma",
        "vmin": -1.0000,
        "vmax": 1.0000,
        "derived": True,
    },
    "bvec": {
        "title": r"$B$",
        "cmap": "RdBu",
        "vsymmetric": True,
        "vmin": -1.0000,
        "vmax": 1.0000,
        "derived": True,
    },
}

def get_normalization(var : str, conf : Configuration):
    """Get the normalization factor for a given variable; simulation-specific
    """
    norm = 1.0
    n0 = 2.0 * conf.ppc * conf.stride**3  # number density per pixel in n_0
    qe = np.abs(conf.qe)
    me_per_qe = np.abs(conf.me) / qe  # for electrons = 1
    deltax = 1.0 / conf.c_omp  # \Delta x in units of skin depth

    lenscale = (
        conf.Nx * conf.NxMesh * deltax / conf.max_mode
    )  # (large-eddy size in units of skin depth)

    if var == "rho":  # or var == 'dens':
        norm = qe * n0
    if var == "dens":
        norm = n0
    if var == "jz":
        norm = qe * n0 * conf.cfl**2
    if var in ["bx", "by", "bz"]:
        norm = conf.binit
    if var == "je":
        norm_E = (me_per_qe * conf.cfl**2) / deltax / lenscale
        norm_J = qe * n0 * conf.cfl**2
        norm = norm_E * norm_J

        # correct for stride size in e/b fields
        # norm /= conf.stride**2
        norm /= 1.0e3

    return norm


def read_full_box(outdir, fname_fld, var, lap):
    """Read hdf5 data of the entire domain
    """
    fields_file = outdir + "/" + fname_fld + "_" + str(lap) + ".h5"
    f5 = h5.File(fields_file, "r")
    return pytools.read_h5_array(f5, var)

def get_smaller_domain(data_array : np.ndarray, new_width : int, start_index : int) -> np.ndarray:
    """Get a smaller domain from a full box simulation to save computational time

    Args:
        data_array (numpy 3d array): cubic box of some data
        new_width (int): width of the smaller domain in number of cells
        start_index (int): starting index of the bigger domain

    Returns:
        numpy 3d array: cropped cubic box
    """
    if start_index + new_width > data_array.shape[0]:
        print("Cannot crop, smaller domain is outside of current domain")
        return 
    else:
        return data_array[start_index:start_index+new_width, start_index:start_index+new_width, start_index:start_index+new_width]
    
def pseudo_convolution(variable : np.ndarray, kernel : np.ndarray, step_size : int) -> np.ndarray:
    """Pseudo-convolution of a variable with a kernel

    Args:
        variable (numpy 3d array): the variable to be convolved
        kernel (numpy 3d array): nxnxn kernel to convolve with
        step_size (int): smoothing factor

    Returns:
        numpy 3d array: the convolved variable
    """
    average_of_each_window = np.zeros(shape=([int(variable.shape[0]/step_size), int(variable.shape[1]/step_size), int(variable.shape[2]/step_size)]))
    pseudo_value = np.zeros(shape=variable.shape)

    # for a set step_size, compute the mean of each step_size x step_size x step_size window within the simulation
    average_of_each_window = zoom(input=variable, zoom = 1./step_size)

    # compute the convolution between average_of_each_window and kernel
    convolved_window = convolve(average_of_each_window, kernel, mode="wrap", cval=0.0)

    for i in range(convolved_window.shape[0]):
        for j in range(convolved_window.shape[1]):
            for k in range(convolved_window.shape[2]):
                pseudo_value[i*step_size:(i+1)*step_size, j*step_size:(j+1)*step_size, k*step_size:(k+1)*step_size] = convolved_window[i,j,k]

    return pseudo_value


def convolve_by_slice(variable_slice : np.ndarray, k : int, kernelA : np.ndarray, kernelS : np.ndarray, direction : np.ndarray) -> np.ndarray:
    """Convolve a 3D variable with an antisymmetric kernel and a symmetric kernel by 2D slices

    Args:
        variable (numpy 2d array): the slice of variable to be convolved; already extended to size + kn to take edges into account
        k (int): slice number
        kernelA (numpy 2d array): nxn antisymmetric kernel to convolve with; normalize beforehand
        kernelS (numpy 2d array): nxn symmetric kernel to convolve with; normalize beforehand
        direction (numpy 4d array): direction to align the kernel with; should already been normalized such that all cell contains a unit vector

    Returns:
        numpy 3d array: the convolved variable with the antisymmetric kernel [index 0] and symmetric kernel [index 1]
    """
    kernel_size = kernelA.shape[0]
    actual_variable_size0 = variable_slice.shape[0] - kernel_size + 1
    actual_variable_size1 = variable_slice.shape[1] - kernel_size + 1
    convolved_variable_slice = np.zeros((2, actual_variable_size0, actual_variable_size1))
    
    print("Convolution on slice {}".format(k), flush=True)

    for j in range(actual_variable_size1):
        for i in range(actual_variable_size0):
            n = direction[:,k,j,i]
            rotate_angle = np.arctan2(n[2], n[1]) * 180 / np.pi - 90
            kernelA_rotated = rotate(kernelA, rotate_angle, reshape=False, mode="nearest", order=1)
            kernelS_rotated = rotate(kernelS, rotate_angle, reshape=False, mode="nearest", order=1)
            
            convolved_variable_slice[0,j,i] = np.einsum("ji,ji", variable_slice[j:j+kernel_size,i:i+kernel_size], kernelA_rotated)
            convolved_variable_slice[1,j,i] = np.einsum("ji,ji", variable_slice[j:j+kernel_size,i:i+kernel_size], kernelS_rotated)

    return convolved_variable_slice

def obtain_convolved_features(feature_array : np.ndarray, kernelA : np.ndarray, kernelS : np.ndarray, direction : np.ndarray) -> np.ndarray:
    """Obtain the convolved features for a 3D array of features with multiprocessing

    Args:
        feature_array (numpy 3d array): the 3D array of features to be convolved
        kernelA (numpy 2d array): nxn antisymmetric kernel to convolve with; normalize beforehand
        kernelS (numpy 2d array): nxn symmetric kernel to convolve with; normalize beforehand
        direction (numpy 4d array): direction to align the kernel with; should already been normalized such that all cell contains a unit vector

    Returns:
        numpy 3d array: the convolved features with the antisymmetric kernel and symmetric kernel
    """
    convolved_feature = np.zeros((2, feature_array.shape[0], feature_array.shape[1], feature_array.shape[2]))

    kernel_size = kernelA.shape[0]
    extended_feature = np.pad(feature_array, ((kernel_size//2,kernel_size//2),(kernel_size//2,kernel_size//2),(kernel_size//2,kernel_size//2)), mode='wrap')
    print("Computing directional convolution using {} cores".format(cpu_count()), flush=True)

    with Pool() as p:
        items = [(extended_feature[k + kernel_size//2], k, kernelA, kernelS, direction) for k in range(feature_array.shape[0])]

        for k, result in enumerate(p.starmap(convolve_by_slice, items)):
            convolved_feature[:, k, :, :] = result

    convolvedA_feature = convolved_feature[0]
    convolvedS_feature = convolved_feature[1]
        
    return convolvedA_feature, convolvedS_feature


def GetH5FileName(feature_list : list[str], snapshot : int, kwargs=[]) -> str:
    """Generate the name of the hdf5 file to save the feature set

    Args:
        feature_list (list[str]): list of features in the saved file
        snapshot (int): snapshot number
        kwargs (list[str], optional): additional strings to add to the end of the name. Defaults to empty.

    Returns:
        str: name of the new hdf5 file containing the feature set
    """
    number_of_rhos = 0
    number_of_js = 0
    number_of_bs = 0
    number_of_es = 0
    for i in feature_list:
        if i[0] == "r":
            number_of_rhos += 1
        elif i[0] == "j":
            number_of_js += 1
        elif i[0] == "b":
            number_of_bs += 1
        elif i[0] == "e":
            number_of_es += 1
    
    if len(kwargs) > 0:
        str_kwargs = ""
        for key in kwargs:
            str_kwargs += "_{}".format(key)
        return "features_{}j{}b{}e{}r_{}{}.h5".format(number_of_js, number_of_bs, number_of_es, number_of_rhos, snapshot, str_kwargs)
    else:
        return "features_{}j{}b{}e{}r_{}.h5".format(number_of_js, number_of_bs, number_of_es, number_of_rhos, snapshot)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='obtain the feature set for SOM training with this script')
    parser.add_argument("--conf_path", type=str, dest='configuration_path', default='/mnt/ceph/users/tha10/sim_snapshots/')
    parser.add_argument("--conf_name", type=str, dest='configuration_name', default='x16.ini')
    parser.add_argument("--generic_file", dest='generic_file', action='store_true', help="Use generic file shape, made specifically for Harris sheet simulations")
    parser.add_argument("--snapshot", type=int, dest='lap', default=2800)
    parser.add_argument("--scale", dest='scale', action='store_true', help="Scale the dataset")
    parser.add_argument("--b_guide", type=str, dest='guide', default='z') # guide field direction
    parser.add_argument("--crop", type=tuple, dest='crop', default=None)
    parser.add_argument("--downsample", type=int, dest='downsample', default=1)
    parser.add_argument("-v", "--var", nargs='+', action="store", type=str, dest='var', default=['j_sym', 'j_asym'])

    args_cli = parser.parse_args()

    do_print = False
    lap = args_cli.lap
    
    if not args_cli.generic_file:
        os.chdir(args_cli.configuration_path)

        # args_cli = pytools.parse_args()
        conf_filename = args_cli.configuration_path + args_cli.configuration_name
        conf = Configuration(conf_filename, do_print=do_print)
        var = "jpar"  # manually set the plotted variable

        # general defaults
        args = {}
        for key in default_values:
            args[key] = default_values[key]

        # overwrite with turbulence defaults
        try:
            for key in default_turbulence_values[var]:
                args[key] = default_turbulence_values[var][key]
        except:
            pass

        print("File location : ", conf.outdir)
        # print("plotting {}".format(var))

        fname_fld = args["file"]
        fname_prtcls = "test-prtcls"

        # --------------------------------------------------
        print(fname_fld)
        rho = read_full_box(conf.outdir, fname_fld, "rho", lap).T

        jx = read_full_box(conf.outdir, fname_fld, "jx", lap).T
        jy = read_full_box(conf.outdir, fname_fld, "jy", lap).T
        jz = read_full_box(conf.outdir, fname_fld, "jz", lap).T

        bx = read_full_box(conf.outdir, fname_fld, "bx", lap).T
        by = read_full_box(conf.outdir, fname_fld, "by", lap).T
        bz = read_full_box(conf.outdir, fname_fld, "bz", lap).T

        ex = read_full_box(conf.outdir, fname_fld, "ex", lap).T
        ey = read_full_box(conf.outdir, fname_fld, "ey", lap).T
        ez = read_full_box(conf.outdir, fname_fld, "ez", lap).T

        # normalize
        rho /= get_normalization("rho", conf)

        jx /= get_normalization("jx", conf)
        jy /= get_normalization("jy", conf)
        jz /= get_normalization("jz", conf)

        bx /= get_normalization("bx", conf)
        by /= get_normalization("by", conf)
        bz /= get_normalization("bz", conf)

        ex /= get_normalization("ex", conf)
        ey /= get_normalization("ey", conf)
        ez /= get_normalization("ez", conf)
        
        dx = conf.stride / conf.c_omp  # skindepth resolution
        
        # rho is weirdly distributed, so we need to do some preprocessing
        rho = read_full_box(conf.outdir, fname_fld, "rho", lap).T
        rho /= get_normalization("rho", conf)
        rho = rho.clip(0.0, 18000)
    else:
        os.chdir(args_cli.configuration_path)
        filename = args_cli.configuration_path + "harris_sheet_00{}.h5".format(str(args_cli.lap))
        print("File location : ", filename)
        
        f_out = h5.File(filename, "r")
        
        jx = f_out["jx"][()]
        jy = f_out["jy"][()]
        jz = f_out["jz"][()]
        
        bx = f_out["bx"][()]
        by = f_out["by"][()]
        bz = f_out["bz"][()]
        
        ex = f_out["ex"][()]
        ey = f_out["ey"][()]
        ez = f_out["ez"][()]
        
        rho = f_out["rho"][()]
        
        dx = 1.
        

    print("Shape of the datacube : ", np.shape(rho), flush=True)
    nz, ny, nx = np.shape(rho)

    
    origin = 0, 0, 0

    Lx = nx * dx
    Ly = ny * dx
    Lz = nz * dx

    midx = Lx // 2
    midy = Ly // 2
    midz = Lz // 2
    
    # magnitude components
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    j_mag = np.sqrt(jx**2 + jy**2 + jz**2)
    e_mag = np.sqrt(ex**2 + ey**2 + ez**2)

    # perpendicular components
    if args_cli.guide == 'z':
        b_perp = np.sqrt(bx**2 + by**2)
    elif args_cli.guide == 'x':
        b_perp = np.sqrt(by**2 + bz**2)
    elif args_cli.guide == 'y':
        b_perp = np.sqrt(bx**2 + bz**2)

    # dot products
    b_vec = np.array([bx, by, bz])
    j_vec = np.array([jx, jy, jz])
    e_vec = np.array([ex, ey, ez]) # this give a 3 x nx x ny x nz 4D array

    b_dot_j = np.einsum('nijk,nijk->ijk', b_vec, j_vec) # dot product performed over each cell; this is much faster than list comprehension and more easily understood
    b_dot_e = np.einsum('nijk,nijk->ijk', b_vec, e_vec)
    e_dot_j = np.einsum('nijk,nijk->ijk', e_vec, j_vec)
    
    print("Finished computing base features", flush=True)
    
    j_par = b_dot_j / b_mag
    j_par_abs = np.abs(j_par)
    j_perp2 = j_mag*j_mag - j_par*j_par 
    j_perp2[j_perp2 < 0.] = 0. # there is a small negative residual before taking the square root, so reset the negative values to zero
    j_perp = np.sqrt(j_perp2)
    e_par = b_dot_e / b_mag
    e_perp = np.sqrt(e_mag*e_mag - e_par*e_par)
    
    print("Finished computing derived features", flush=True)
    
    list_of_features = args_cli.var
    
    # isotropic convolution
    smoothing_convolution_on = False
    for name in list_of_features:
        if "ciso" in name:
            smoothing_convolution_on = True
            print("Isotropic convolution requested", flush=True)
            break
            
    if smoothing_convolution_on:
        convolution_size = 4
        kernel_iso_shape = (3,3,3)
        kernel_iso = np.ones(shape=kernel_iso_shape)
        kernel_iso /= np.sum(kernel_iso)
        
        j_mag_ciso = pseudo_convolution(j_mag, kernel_iso, step_size=convolution_size)
        j_par_ciso = pseudo_convolution(j_par, kernel_iso, step_size=convolution_size)
        j_par_abs_ciso = pseudo_convolution(j_par_abs, kernel_iso, step_size=convolution_size)
        j_perp_ciso = pseudo_convolution(j_perp, kernel_iso, step_size=convolution_size)
        e_par_ciso = pseudo_convolution(e_par, kernel_iso, step_size=convolution_size)
        e_perp_ciso = pseudo_convolution(e_perp, kernel_iso, step_size=convolution_size)
        bz_ciso = pseudo_convolution(bz, kernel_iso, step_size=convolution_size)
        b_perp_ciso = pseudo_convolution(b_perp, kernel_iso, step_size=convolution_size)
        e_dot_j_ciso = pseudo_convolution(e_dot_j, kernel_iso, step_size=convolution_size)
        rho_ciso = pseudo_convolution(rho, kernel_iso, step_size=convolution_size)
        
        print("Finished computing isotropic convolution features", flush=True)

    # antisymmetric and symmetric convolution
    directional_convolution_on = False
    for name in list_of_features:
        if "sym" in name:
            directional_convolution_on = True
            print("Directional convolution requested", flush=True)
            break
    
    if directional_convolution_on:
        # define the kernels
        kn=19
        along_k, across_k = np.linspace(-1, 1, kn), np.linspace(-1, 1, kn)

        coord_kA = np.asarray(np.meshgrid(along_k, across_k))

        kernelA = np.zeros((kn,kn))
        for i in range(kn):
            for j in range(kn):
                if i != kn//2 or j != kn//2:
                    hypot_ij = np.hypot(coord_kA[0,i,j], coord_kA[1,i,j])
                    kernelA[i,j] = -coord_kA[0,i,j]/hypot_ij if hypot_ij <= 1 - 1/kn else 0
        kernelA /= np.sum(np.abs(kernelA))

        kernelS = np.zeros((kn,kn))
        kernelS[kn//2,:] = 1

        for i in range(kn):
            for j in range(kn):
                if i != kn//2 or j != kn//2:
                    hypot_ij = np.hypot(coord_kA[0,i,j], coord_kA[1,i,j])
                    kernelS[i,j] = np.abs(coord_kA[0,i,j]/hypot_ij) if hypot_ij <= 1 - 1/kn else 0
        kernelS /= np.sum(np.abs(kernelS))

        # calculate the direction vector (grad J)
        grad_jpar = np.gradient(j_par)
        grad_jpar = np.array([grad_jpar[2], grad_jpar[1], grad_jpar[0]])
        grad_jpar_hat = grad_jpar / np.linalg.norm(grad_jpar, axis=0)

        print(f"Shape of j_par : {j_par.shape}")
        j_asym, j_sym = obtain_convolved_features(j_par, kernelA, kernelS, grad_jpar_hat)
        

        print("Finished computing directional convolution features", flush=True)
    
    ### FEATURE SET ###
    dataset_combined = np.array([globals()[list_of_features[i]] for i in range(len(list_of_features))])
    #######################################################
    print("Dataset contains : ",list_of_features, flush=True)
    
    additional_kws = []
    
    if "j_mag" in list_of_features:
        additional_kws.append("jmag")
    if "j_par_abs" in list_of_features:
        additional_kws.append("jpa")
    if args_cli.guide != 'z':
        additional_kws.append("bg" + args_cli.guide)
    if "j_asym" in list_of_features:
        additional_kws.append("jasym")
    if "j_sym" in list_of_features:
        additional_kws.append("jsym")
    
    # save memory by carving out a smaller domain
    if args_cli.crop is not None and nx == ny == nz: 
        start_index_crop = args_cli.crop[0]
        width_of_new_domain = args_cli.crop[1]
        dataset_reduced = np.zeros((dataset_combined.shape[0], width_of_new_domain, width_of_new_domain, width_of_new_domain))
        for i in range(dataset_combined.shape[0]):
            dataset_reduced[i,:,:,:] = get_smaller_domain(dataset_combined[i,:,:,:], new_width=width_of_new_domain, start_index=start_index_crop)
        additional_kws.append(f'c{start_index_crop}-{width_of_new_domain}')
    else:
        start_index_crop = 0
        width_of_new_domain = nx
        dataset_reduced = dataset_combined
    
    # downsample the dataset
    if args_cli.downsample > 1: 
        start_index_crop = 0
        dataset_reduced = dataset_reduced[:,::args_cli.downsample,::args_cli.downsample,::args_cli.downsample]
        width_of_new_domain = dataset_reduced.shape[1]
        additional_kws.append(f'd{args_cli.downsample}')
        print("Downsampled dataset by factor of {}".format(args_cli.downsample), flush=True)


    dataset_reshaped = np.reshape(dataset_reduced, (dataset_reduced.shape[0], dataset_reduced.shape[1]*dataset_reduced.shape[2]*dataset_reduced.shape[3])).T

    if args_cli.scale == True: # scale dataset
        scaler = MinMaxScaler()
        scaler.fit(dataset_reshaped)
        dataset_scaled = scaler.transform(dataset_reshaped)
    else:
        dataset_scaled = dataset_reshaped
         
    # save the dataset
    f5 = h5.File(GetH5FileName(list_of_features, lap, kwargs=additional_kws), 'w')
    dsetx = f5.create_dataset("features",  data=dataset_scaled)

    asciilist = [n.encode("ascii", "ignore") for n in list_of_features]
    dsetf = f5.create_dataset("names",  data=asciilist)
    f5.close()
    print("Saved dataset to {}".format(GetH5FileName(list_of_features, lap, kwargs=additional_kws)), flush=True)
    print("Order of features : ", list_of_features, flush=True)