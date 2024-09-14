### This script is used to read the h5 file and plot the data from turbulence box simulation

import h5py as h5
import numpy as np

import os
import sys

os.chdir('/mnt/home/tha10/ceph/sim_snapshots/')
sys.path.append('/mnt/home/tha10/ceph/sim_snapshots/')

import pytools
from pytools.pybox.box import Box
from pytools.pybox.box import axisEqual3D

from scipy.ndimage import map_coordinates, rotate, convolve
from matplotlib.backends.backend_pdf import PdfPages

from initialize_turbulence import Configuration_Turbulence as Configuration

# f_in = h5.File("/mnt/home/tha10/ceph/sim_snapshots/d3x128s10/flds_5000.h5","r")
# list(f_in.keys())

# conf = "d3x128s10.ini"
# path = "/mnt/home/tha10/ceph/sim_snapshots/d3x128s10/"
# filename = "flds_5000.h5"
# lap = 5000

class GetH5Data:
    def __init__(self, path, filename, conf, lap):
        do_print = False

        self.f_in = h5.File(path + filename, "r")
        self.keys = list(self.f_in.keys())
        self.conf = Configuration(conf, do_print=do_print)

        self.default_values = {
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


        self.default_turbulence_values = {
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

        var = "j_par"  # manually set the plotted variable

        # general defaults
        args = {}
        for key in self.default_values:
            args[key] = self.default_values[key]

        # overwrite with turbulence defaults
        try:
            for key in self.default_turbulence_values[var]:
                args[key] = self.default_turbulence_values[var][key]
        except:
            pass

        print(self.conf.outdir)
        print("plotting {}".format(var))

        self.fname_fld = args["file"]
        self.fname_prtcls = "test-prtcls"

        self.lap = lap
        self.do_print = False


    def get_normalization(self, var):
        norm = 1.0
        n0 = 2.0 * self.conf.ppc * self.conf.stride**3  # number density per pixel in n_0
        qe = np.abs(self.conf.qe)
        me_per_qe = np.abs(self.conf.me) / qe  # for electrons = 1
        deltax = 1.0 / self.conf.c_omp  # \Delta x in units of skin depth

        lenscale = (
            self.conf.Nx * self.conf.NxMesh * deltax / self.conf.max_mode
        )  # (large-eddy size in units of skin depth)

        if var == "rho":  # or var == 'dens':
            norm = qe * n0
        if var == "dens":
            norm = n0
        if var == "jz":
            norm = qe * n0 * self.conf.cfl**2
        if var in ["bx", "by", "bz"]:
            norm = self.conf.binit
        if var == "je":
            norm_E = (me_per_qe * self.conf.cfl**2) / deltax / lenscale
            norm_J = qe * n0 * self.conf.cfl**2
            norm = norm_E * norm_J

            # correct for stride size in e/b fields
            # norm /= conf.stride**2
            norm /= 1.0e3

        return norm


    def read_full_box(self, var):
        fields_file = self.conf.outdir + "/" + self.fname_fld + "_" + str(self.lap) + ".h5"
        f5 = h5.File(fields_file, "r")
        return pytools.read_h5_array(f5, var)

    def load_som_cluster(self, path):
        f_out = h5.File(path, "r")
        return np.array(f_out['cluster_id'][()])

    # reading function for data
    def read_h5(self, var):
        return self.read_full_box(var)

    # --------------------------------------------------

    def return_basic_j_fields(self, dx):
        print(self.fname_fld)
        self.dx = dx
        self.rho = self.read_h5("rho").T

        self.jx = self.read_h5("jx").T
        self.jy = self.read_h5("jy").T
        self.jz = self.read_h5("jz").T

        # normalize
        self.rho /= self.get_normalization("rho")

        self.jx /= self.get_normalization("jx")
        self.jy /= self.get_normalization("jy")
        self.jz /= self.get_normalization("jz")

        self.jz_rms = np.sqrt(np.mean(self.jz**2))

        self.nx, self.ny, self.nz = np.shape(self.rho)

        # return self.rho, self.jx, self.jy, self.jz
    
    def return_all_other_fields(self):
        self.bx = self.read_h5("bx").T
        self.by = self.read_h5("by").T
        self.bz = self.read_h5("bz").T
        
        self.ex = self.read_h5("ex").T
        self.ey = self.read_h5("ey").T
        self.ez = self.read_h5("ez").T


        self.bx /= self.get_normalization("bx")
        self.by /= self.get_normalization("by")
        self.bz /= self.get_normalization("bz")

        self.ex /= self.get_normalization("ex")
        self.ey /= self.get_normalization("ey")
        self.ez /= self.get_normalization("ez")

        # magnitude components
        self.b_mag = np.sqrt(self.bx**2 + self.by**2 + self.bz**2)
        self.j_mag = np.sqrt(self.jx**2 + self.jy**2 + self.jz**2)
        self.e_mag = np.sqrt(self.ex**2 + self.ey**2 + self.ez**2)

        # perpendicular component of B
        self.b_perp = np.sqrt(self.bx**2 + self.by**2)

        # dot products
        self.b_vec = np.array([self.bx, self.by, self.bz])
        self.j_vec = np.array([self.jx, self.jy, self.jz])
        self.e_vec = np.array([self.ex, self.ey, self.ez]) # this give a 3 x nx x ny x nz 4D array

        self.b_dot_j = np.einsum('nijk,nijk->ijk', self.b_vec, self.j_vec) # dot product performed over each cell; this is much faster than list comprehension and more easily understood
        self.b_dot_e = np.einsum('nijk,nijk->ijk', self.b_vec, self.e_vec)
        self.e_dot_j = np.einsum('nijk,nijk->ijk', self.e_vec, self.j_vec)
        self.j_par = self.b_dot_j / self.b_mag
        self.j_perp2 = self.j_mag * self.j_mag - self. j_par * self.j_par # there is a small negative residual before taking the square root, so reset the negative values to zero
        self.j_perp2[self.j_perp2 < 0.] = 0.
        self.j_perp = np.sqrt(self.j_perp2)
        self.e_par = self.b_dot_e / self.b_mag
        self.e_perp = np.sqrt(self.e_mag*self.e_mag - self.e_par*self.e_par)

        self.jpar_rms = np.sqrt(np.mean(self.j_par**2))
        # print("jpar rms: ", jpar_rms)
        # j_par[np.abs(j_par) < jpar_rms] = 0.

    def downsample(self, arr, factor):
        return arr[::factor, ::factor, ::factor]

    def downsample_basic_j_fields(self, factor):
        self.dx *= factor
        self.rho = self.downsample(self.rho, factor)
        self.jx = self.downsample(self.jx, factor)
        self.jy = self.downsample(self.jy, factor)
        self.jz = self.downsample(self.jz, factor)

        self.nx, self.ny, self.nz = np.shape(self.rho)
        
    def downsample_all_other_fields(self, factor):
        self.bx = self.downsample(self.bx, factor)
        self.by = self.downsample(self.by, factor)
        self.bz = self.downsample(self.bz, factor)

        self.ex = self.downsample(self.ex, factor)
        self.ey = self.downsample(self.ey, factor)
        self.ez = self.downsample(self.ez, factor)

        self.b_vec = np.array([self.bx, self.by, self.bz])
        self.j_vec = np.array([self.jx, self.jy, self.jz])
        self.e_vec = np.array([self.ex, self.ey, self.ez]) # this give a 3 x nx x ny x nz 4D array

        self.b_mag = self.downsample(self.b_mag, factor)
        self.j_mag = self.downsample(self.j_mag, factor)
        self.e_mag = self.downsample(self.e_mag, factor)

        self.b_perp = self.downsample(self.b_perp, factor)
        self.b_dot_j = self.downsample(self.b_dot_j, factor)
        self.b_dot_e = self.downsample(self.b_dot_e, factor)
        self.e_dot_j = self.downsample(self.e_dot_j, factor)
        self.j_par = self.downsample(self.j_par, factor)
        self.j_perp = self.downsample(self.j_perp, factor)
        self.e_par = self.downsample(self.e_par, factor)
        self.e_perp = self.downsample(self.e_perp, factor)
    