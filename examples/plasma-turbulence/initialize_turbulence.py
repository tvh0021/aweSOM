from __future__ import print_function

from pytools import Configuration

import numpy as np
from numpy import sqrt, pi


# extend default conf class with problem specific parameters
class Configuration_Turbulence(Configuration):

    def __init__(self, *file_names, do_print=False):
        Configuration.__init__(self, *file_names)

        # problem specific initializations
        if do_print:
            print("Initializing turbulence setup...")

        # local variables just for easier/cleaner syntax
        me = np.abs(self.me)
        mi = np.abs(self.mi)
        c = self.cfl
        ppc = self.ppc * 2.0  # multiply x2 to account for 2 species/pair plasma

        # plasma reaction & subsequent normalization
        self.gamma = 1.0
        self.omp = c / self.c_omp
        self.qe = -(self.omp ** 2.0 * self.gamma) / ((ppc * 0.5) * (1.0 + me / mi))
        self.qi = -self.qe

        me *= abs(self.qi)
        mi *= abs(self.qi)

        # temperatures
        self.delgam_e = self.delgam
        self.delgam_i = self.delgam


        # ---------cold plasma-----------
        # parse external magnetic field strength from sigma_ext

        # determine initial magnetic field based on magnetization sigma which
        # is magnetic energy density/ kinetic energy density
        # this definition works even for nonrelativistic flows.
        #self.binit = sqrt(
        #    (self.gamma) * ppc * 0.5 * c ** 2.0 * (me * (1.0 + me / mi)) * self.sigma
        #)

        # no corrections; cold sigma
        self.binit_nc = sqrt(ppc*c**2.*self.sigma*me)

        # ---------hot plasma-----------
        delgam_i = self.delgam_i
        delgam_e = self.delgam_e

        corrdelgam_qe  = self.delgam_e
        corrdelgam_sig = self.delgam_e

        zeta=delgam_i/(0.24 + delgam_i)
        gad_i=1./3.*(5 - 1.21937*zeta + 0.18203*zeta**2 - 0.96583*zeta**3 + 2.32513*zeta**4 - 2.39332*zeta**5 + 1.07136*zeta**6)
        delgam_e=self.delgam*mi/me*self.temperature_ratio 
        zeta=delgam_e/(0.24 + delgam_e)
        gad_e=1./3.*(5 - 1.21937*zeta + 0.18203*zeta**2 - 0.96583*zeta**3 + 2.32513*zeta**4 - 2.39332*zeta**5 + 1.07136*zeta**6)

        #hot pair-plasma correction factor
        self.warm_corr = 0.5*( \
                         (1.0 + corrdelgam_sig*gad_i/(gad_i-1.)*self.delgam_i) \
                        +(1.0 + corrdelgam_sig*gad_e/(gad_e-1.)*self.delgam_e) \
                             )

        self.binit_warm = sqrt(ppc*.5*c**2.* \
                (mi*(1.+corrdelgam_sig*gad_i/(gad_i-1.)*self.delgam_i)+me*(1.+ \
                corrdelgam_sig*gad_e/(gad_e-1.)*self.delgam_e))*self.sigma)


        # approximative \gamma_th = 1 + 3\theta
        self.gammath = 1.0 + 3.0*delgam_e
        self.binit_approx = sqrt(self.gammath*ppc*me*c**2.*self.sigma)

        #manual value for theta=0.3
        #self.gammath = 1.55 

        self.gammath = 1.0 # cool plasma
        self.binit = sqrt(self.gammath*ppc*me*c**2.*self.sigma)

        self.sigma_perp = self.sigma*self.drive_ampl**2
        self.binit_perp = sqrt(self.gammath*ppc*me*c**2.*self.sigma_perp)

        if do_print:
            print("init: sigma: ", self.sigma)
            print("init: mass term: ", sqrt(mi+me))
            print("init: warm corr:: ", self.warm_corr)
            print("init: B_guide (manual): ", self.binit)
            print("init: B_guide (no corrections): ", self.binit_nc)
            print("init: B_guide (approx): ", self.binit_approx)
            print("init: B_guide (warm): ", self.binit_warm)
            print("init: q_e: ", self.qe)
            print("init: q_i: ", self.qi)


        #-------------------------------------------------- 
        # radiation drag, if any

        if "gammarad" in self.__dict__:
            if not(self.gammarad == 0.0):
                #self.drag_amplitude = 0.1*self.binit*np.abs(self.qe)/(self.gammarad**2.0)
                #self.drag_amplitude = 0.1*self.binit/(self.gammarad**2.0)
                #self.drag_amplitude = 0.1*self.binit_perp/(self.gammarad**2.0)
                self.drag_amplitude = 0.1*self.binit/(self.gammarad**2.0)

                if do_print:
                    print("init: using radiation drag...")
                    print("init:  drag amplitude: {} with gamma_rad: {}".format(self.drag_amplitude, self.gammarad))
        else:
            self.gammarad = 0.0

        if "radtemp" in self.__dict__:
            if not(self.radtemp == 0.0):
                if do_print:
                    print("init: using radiation drag with temperature...")
                    print("init:  drag temperature: {}".format(self.radtemp))
        else:
            self.radtemp = 0.0


        #forcing scale in units of skin depth
        self.l0 = self.Nx*self.NxMesh/self.max_mode/self.c_omp  
        
        #thermal larmor radius
        lth = self.gammath / np.sqrt(self.sigma_perp)*np.sqrt(self.gammath) #gamma_0

        #reconnection radius; gam ~ sigma
        lsig = self.sigma_perp / np.sqrt(self.sigma_perp)*np.sqrt(self.gammath) #gamma_0

        #self.g0 = self.l0*np.sqrt(self.sigma)*np.sqrt(self.gammath) #gamma_0
        #A = (0.1*3.0/4.0)*self.g0/self.gammarad**2 #definition we use in Runko

        self.g0 = self.l0*np.sqrt(self.sigma_perp)*np.sqrt(self.gammath) #gamma_0
        A = 0.1*self.g0/self.gammarad**2 #definition we use in Runko

        if do_print:
            print("init: l_0:", self.l0)
            print("init: l_th:", lth)
            print("init: l_sig:", lsig)
            print("init: gamma_0: ", self.g0)
            print("init: A = {}".format(A))
            print("init: gcool = {}".format(1/A))
            print("init: gcrit = {}".format(self.gammarad))
            print("init: sigm0 = {}".format(self.sigma))
            print("init: sigmp = {}".format(self.sigma_perp))


        #running time estimates
        lx = self.Nx*self.NxMesh/self.max_mode
        t0 = lx/self.cfl/self.drive_ampl**2

        if do_print:
            print("init: lap(t = 5  l_0/c):",  5*t0)
            print("init: lap(t = 10 l_0/c):", 10*t0)
            print("init: lap(t = 20 l_0/c):", 20*t0)
            print("init: sampling rate:",  self.interval/t0)


