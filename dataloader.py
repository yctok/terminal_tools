import numpy as np
import matplotlib.pyplot as plt

import aurora
import netCDF4 as nc
import glob

import sys
sys.path.append('/nobackup1/millerma/solps-runs/SOLPSxport')
import SOLPSutils as sut

from scipy.interpolate import interp1d
from scipy import interpolate
import os


class experiment:

    def __init__(self, pshot, gshot):

        self.pshot = pshot
        self.gshot = gshot
    
    def populate(self):

        if self.pshot == '1070614013':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070614013_1020_1060_SOLPS.pkl'
        elif self.pshot == '1070614015':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070614015_1140_1175_SOLPS.pkl'
        elif self.pshot == '1070614016':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070614016_1000_1030_SOLPS.pkl'
        elif self.pshot == '1070821008':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070821008_1000_1280_SOLPS.pkl'
        elif self.pshot == '1070525013':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070525013_800_1000_SOLPS.pkl'
        elif self.pshot == '1070821004':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070821004_900_1050_SOLPS.pkl'
        elif self.pshot == '1070821003':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070821003_1150_1450_SOLPS.pkl'
        elif self.pshot == '1070821009':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070821009_980_1080_SOLPS.pkl'
        elif self.pshot == '1070821013':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070821013_900_950_SOLPS.pkl'
        elif self.pshot == '1070821023':
            arr_file = '/nobackup1/millerma/solps-runs/exp_data/lyman_data_1070821023_1400_1480_SOLPS.pkl'

        import pickle as pkl
        with open(arr_file,'rb') as f:
            res = pkl.load(f)

        ## plasma
    
        # fits
        self.psin = res['rhop_fit']**2
        self.R = res['R_fit']
        self.ne = res['ne_fit']*1e6 # cm^-3 --> m^-3
        self.ne_unc = res['ne_unc_fit']*1e6 # cm^-3 --> m^-3
        self.Te = res['Te_fit']
        self.Te_unc = res['Te_unc_fit'] 
    
        # raw
        self.psin_TS = res['rhop_raw']**2
        self.ne_TS = res['ne_raw']*1e6 # cm^-3 --> m^-3
        self.ne_unc_TS = res['ne_unc_raw']*1e6 # cm^-3 --> m^-3
        self.Te_TS = res['Te_raw'] # eV 
        self.Te_unc_TS = res['Te_unc_raw'] # eV 


        # double check units for the raw points here still! 
        ## neutrals

        # fits
        self.n0 = res['nn_fit']*1e6 # cm^-3 --> m^-3
        self.n0_unc = res['nn_unc_fit']*1e6 # cm^-3 --> m^-3
        self.S_ion = res['S_ion_fit']*1e6 # cm^-3 --> m^-3
        self.S_ion_unc = res['S_ion_unc_fit']*1e6 # cm^-3 --> m^-3
        
        # raw
        self.n0_TS = res['nn_raw']*1e6
        self.n0_unc_TS = res['nn_unc_raw']*1e6
        self.S_ion_TS = res['S_ion_raw']*1e6
        self.S_ion_unc_TS = res['S_ion_unc_raw']*1e6


        ## Ly-a data

        # fits
        self.emiss = res['emiss_fit']*1e6 # cm^-3 --> m^-3
        self.emiss_unc = res['emiss_unc_fit']*1e6 # cm^-3 --> m^-3
        
        # raw
        self.emiss_TS = res['emiss_raw']
        self.emiss_unc_TS = res['emiss_unc_raw']

        self.R_tang = res['R_tang']
        self.bright = res['bright'] # cm^-3 --> m^-3
        self.bright_unc = res['bright_unc'] # cm^-3 --> m^-3

        ## div probe data
        self.rho_LO = res['rho_LO']
        self.ne_LO = res['ne_LO']
        self.Te_LO = res['Te_LO']
        self.Js_LO = res['Js_LO']

        self.rho_LI = res['rho_LI']
        self.ne_LI = res['ne_LI']
        self.Te_LI = res['Te_LI']
        self.Js_LI = res['Js_LI']

        self.rho_UO = res['rho_UO']
        self.ne_UO = res['ne_UO']
        self.Te_UO = res['Te_UO']
        self.Js_UO = res['Js_UO']

        self.rho_UI = res['rho_UI']
        self.ne_UI = res['ne_UI']
        self.Te_UI = res['Te_UI']
        self.Js_UI = res['Js_UI']


class SOLPS: 

    def __init__(self, gshot, simtype, case, jxa):
    
        self.gshot = gshot # this should be the shot used to generate the grid, so correct geqdsk is found
        self.simtype = simtype
        self.case = case
        self.jxa = jxa

        self.SOLPSWORK = '/nobackup1/millerma/solps-runs'
        self.fdir = self.SOLPSWORK + '/' + str(self.simtype) + '/' + str(self.case)
        self.bdir = self.SOLPSWORK + '/' + str(self.simtype) + '/baserun'
        #self.fdir = '.'

    def load_grid(self): # taken from SOLPSxport.py

        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'

        dsa, crLowerLeft = sut.B2pl('0 crx writ jxa f.y', wdir = self.fdir)
        # dummy, crLowerRight = B2pl('1 crx writ jxa f.y', wdir = wdir)
        # Only 2 unique psi values per cell, grab 0 and 2
        dummy, crUpperLeft = sut.B2pl('2 crx writ jxa f.y', wdir = self.fdir)  # all x inds are the same
        dummy, czLowerLeft = sut.B2pl('0 cry writ jxa f.y', wdir = self.fdir)
        dummy, czUpperLeft = sut.B2pl('2 cry writ jxa f.y', wdir = self.fdir)
        ncells = len(dummy)
    
        import glob
        geqdsk = glob.glob('{}/{}/baserun/g{}.*'.format(self.SOLPSWORK, self.simtype, self.gshot))[0]
        g = sut.loadg(geqdsk)
        
        self.geqdsk = geqdsk
        self.g = g

#       g = sut.loadg(self.data['gfile_loc'])
        psiN = (g['psirz'] - g['simag']) / (g['sibry'] - g['simag'])

        dR = g['rdim'] / (g['nw'] - 1)
        dZ = g['zdim'] / (g['nh'] - 1)

        gR = []
        for i in range(g['nw']):
            gR.append(g['rleft'] + i * dR)

        gZ = []
        for i in range(g['nh']):
            gZ.append(g['zmid'] - 0.5 * g['zdim'] + i * dZ)

        gR = np.array(gR)
        gZ = np.array(gZ)
        
        R_solps_top = 0.5 * (np.array(crLowerLeft) + np.array(crUpperLeft))
        Z_solps_top = 0.5 * (np.array(czLowerLeft) + np.array(czUpperLeft))

        psiNinterp = interpolate.interp2d(gR, gZ, psiN, kind = 'cubic')

        psi_solps = np.zeros(ncells)
        for i in range(ncells):
            psi_solps_LL = psiNinterp(crLowerLeft[i], czLowerLeft[i])
            psi_solps_UL = psiNinterp(crUpperLeft[i], czUpperLeft[i])
            psi_solps[i] = np.mean([psi_solps_LL,psi_solps_UL])

        self.R = np.array(R_solps_top)
        self.psin = np.array(psi_solps)
        os.chdir(olddir)

    def load_kin_profs(self):

        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'

        rx_, ne_ = sut.B2pl("ne 0 0 sumz writ jxa f.y")
        rx_, te_ = sut.B2pl("te 0 0 sumz writ jxa f.y")
        rx_, ti_ = sut.B2pl("ti 0 0 sumz writ jxa f.y")

        os.chdir(olddir)

        #self.psin = np.array(rx_)
        self.ne = np.array(ne_)
        self.Te = np.array(te_)
        self.Ti = np.array(ti_)


    def load_neutrals(self):

        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'

        ## get neutral density and source at the midplane

        _, n0_ = sut.B2pl("dab2 0 0 sumz writ jxa f.y")
        _, T0_ = sut.B2pl("tab2 0 0 sumz writ jxa f.y")

#       os.chdir(olddir)

        fn = '{}/balance.nc'.format(self.fdir)
        ds = nc.Dataset(fn)

        sna = ds['eirene_mc_papl_sna_bal']
        vol = ds['vol']
        crx = ds['crx']
        cry = ds['cry']

        sna_sum = np.sum(sna,axis=0)
        sna_Dplus_vol = sna_sum[1]/vol
        sna_ = sna_Dplus_vol[:,self.jxa]

        # quick lil thing
        import sys
        #sys.path.append('/nobackup1/millerma/solps-runs/scripts_SOLPS')
        #from b2fplasmf_read import read_b2fplasmf
        #b2f = read_b2fplasmf('/nobackup1/millerma/solps-runs/pbal/match15_unpump/b2fplasmf', 96, 36, 2)
        #sna_plasmf = b2f.b2stbr_sna[:,:,1]/np.array(vol).transpose()
        #self.sna_plasmf = sna_plasmf[self.jxa]


        ## get neutral density and source at the separatrix (also flux!)

        # really only want to do this on closed field lines - these numbers may (will) be different if using 96 rather than 98
        inner_lower_ind = 13
        inner_upper_ind = 35
        outer_upper_ind = 62
        outer_lower_ind = 84

        inner_inds = np.arange(inner_lower_ind, inner_upper_ind+1)
        outer_inds = np.arange(outer_upper_ind, outer_lower_ind+1)

        core_mask_3898 = np.hstack((inner_inds, outer_inds))
        core_mask_3696 = core_mask_3898 - 1 # just need to shift the mask down 1 if using this notation

        self.load_kin_profs()
        grad_ne = np.abs(np.gradient(self.ne, self.R))
        ped_ind = 7 #np.where(grad_ne > np.max(grad_ne)/10)[0][0] -1 # just inside of where the gradient is less than 10x the max
        sep1_in_ind = 14
        sep1_out_ind = 15
        sep2_in_ind = 18
        sep2_out_ind = 19

        _, n0_ped_ = sut.B2pl("dab2 0 0 sumz writ {} f.x".format(ped_ind))
        _, pflux_ped_ = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ {} f.x".format(ped_ind))
        
        _, n0_sep1a_ = sut.B2pl("dab2 0 0 sumz writ {} f.x".format(sep1_in_ind))
        _, n0_sep1b_ = sut.B2pl("dab2 0 0 sumz writ {} f.x".format(sep1_out_ind))
        _, pflux_sep1a_ = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ {} f.x".format(sep1_in_ind))
        _, pflux_sep1b_ = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ {} f.x".format(sep1_out_ind))

        _, n0_sep2a_ = sut.B2pl("dab2 0 0 sumz writ {} f.x".format(sep2_in_ind))
        _, n0_sep2b_ = sut.B2pl("dab2 0 0 sumz writ {} f.x".format(sep2_out_ind))
        _, pflux_sep2a_ = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ {} f.x".format(sep2_in_ind))
        _, pflux_sep2b_ = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ {} f.x".format(sep2_out_ind))

        n0_sep1_ = (np.array(n0_sep1a_) + np.array(n0_sep1b_))/2
        n0_sep2_ = (np.array(n0_sep2a_) + np.array(n0_sep2b_))/2
    
        sna_sep1_ = (sna_Dplus_vol[sep1_in_ind] + sna_Dplus_vol[sep1_out_ind])/2
        sna_sep2_ = (sna_Dplus_vol[sep2_in_ind] + sna_Dplus_vol[sep2_out_ind])/2
        
        pflux_sep1_ = (np.array(pflux_sep1a_) + np.array(pflux_sep1b_))/2
        pflux_sep2_ = (np.array(pflux_sep2a_) + np.array(pflux_sep2b_))/2

        # let's get the interface (separatrix) - and calculate its poloidal angle

        # inner separatrix
        _, crLowerLeft1 = sut.B2pl('0 crx writ {} f.x'.format(sep1_in_ind), wdir = self.fdir)
        _, crUpperLeft1 = sut.B2pl('3 crx writ {} f.x'.format(sep1_in_ind), wdir = self.fdir)
        _, crLowerRight1 = sut.B2pl('1 crx writ {} f.x'.format(sep1_out_ind), wdir = self.fdir)
        _, crUpperRight1 = sut.B2pl('2 crx writ {} f.x'.format(sep1_out_ind), wdir = self.fdir)
        
        _, czLowerLeft1 = sut.B2pl('0 cry writ {} f.x'.format(sep1_in_ind), wdir = self.fdir)
        _, czUpperLeft1 = sut.B2pl('3 cry writ {} f.x'.format(sep1_in_ind), wdir = self.fdir)
        _, czLowerRight1 = sut.B2pl('1 cry writ {} f.x'.format(sep1_out_ind), wdir = self.fdir)
        _, czUpperRight1 = sut.B2pl('2 cry writ {} f.x'.format(sep1_out_ind), wdir = self.fdir)

        cr_sep1 = (np.array(crLowerLeft1) + np.array(crUpperLeft1) + np.array(crLowerRight1) + np.array(crUpperRight1))/4
        cz_sep1 = (np.array(czLowerLeft1) + np.array(czUpperLeft1) + np.array(czLowerRight1) + np.array(czUpperRight1))/4

    
        _, crLowerLeft2 = sut.B2pl('0 crx writ {} f.x'.format(sep2_in_ind), wdir = self.fdir)
        _, crUpperLeft2 = sut.B2pl('3 crx writ {} f.x'.format(sep2_in_ind), wdir = self.fdir)
        _, crLowerRight2 = sut.B2pl('1 crx writ {} f.x'.format(sep2_out_ind), wdir = self.fdir)
        _, crUpperRight2 = sut.B2pl('2 crx writ {} f.x'.format(sep2_out_ind), wdir = self.fdir)
        
        _, czLowerLeft2 = sut.B2pl('0 cry writ {} f.x'.format(sep2_in_ind), wdir = self.fdir)
        _, czUpperLeft2 = sut.B2pl('3 cry writ {} f.x'.format(sep2_in_ind), wdir = self.fdir)
        _, czLowerRight2 = sut.B2pl('1 cry writ {} f.x'.format(sep2_out_ind), wdir = self.fdir)
        _, czUpperRight2 = sut.B2pl('2 cry writ {} f.x'.format(sep2_out_ind), wdir = self.fdir)

        cr_sep2 = (np.array(crLowerLeft2) + np.array(crUpperLeft2) + np.array(crLowerRight2) + np.array(crUpperRight2))/4
        cz_sep2 = (np.array(czLowerLeft2) + np.array(czUpperLeft2) + np.array(czLowerRight2) + np.array(czUpperRight2))/4

        # compute the poloidal angle
        
        R0 = 0.68
        Z0 = 0

        dR_sep1 = cr_sep1 - R0
        dZ_sep1 = cz_sep1 - Z0

        theta_sep1_ = np.degrees(np.arctan(dZ_sep1/dR_sep1))
        quadI = (dR_sep1 > 0) & (dZ_sep1 > 0)
        quadII = (dR_sep1 < 0) & (dZ_sep1 > 0)
        quadIII = (dR_sep1 < 0) & (dZ_sep1 < 0)
        quadIV = (dR_sep1 > 0) & (dZ_sep1 < 0)
        theta_sep1_[quadI] = theta_sep1_[quadI] + 0
        theta_sep1_[quadII] = theta_sep1_[quadII] + 180
        theta_sep1_[quadIII] = theta_sep1_[quadIII] + 180
        theta_sep1_[quadIV] = theta_sep1_[quadIV] + 360
        
        dR_sep2 = cr_sep2 - R0
        dZ_sep2 = cz_sep2 - Z0

        theta_sep2_ = np.degrees(np.arctan(dZ_sep2/dR_sep2))
        quadI = (dR_sep1 > 0) & (dZ_sep1 > 0)
        quadII = (dR_sep1 < 0) & (dZ_sep1 > 0)
        quadIII = (dR_sep1 < 0) & (dZ_sep1 < 0)
        quadIV = (dR_sep1 > 0) & (dZ_sep1 < 0)
        theta_sep2_[quadI] = theta_sep2_[quadI] + 0
        theta_sep2_[quadII] = theta_sep2_[quadII] + 180
        theta_sep2_[quadIII] = theta_sep2_[quadIII] + 180
        theta_sep2_[quadIV] = theta_sep2_[quadIV] + 360

        os.chdir(olddir)

        maxZ = np.max(dZ_sep1)
        minZ = np.min(dZ_sep1)

        theta_uxp_ = theta_sep1_[np.where(dZ_sep1 == maxZ)[0]]
        theta_lxp_ = theta_sep1_[np.where(dZ_sep1 == minZ)[0]]

        self.nR = np.mean(crx[:,:,self.jxa],axis=0)
        self.n0 = np.array(n0_)
        self.T0 = np.array(T0_)
        self.S_at_ion = np.array(sna_)

        self.theta_sep1 = np.array(theta_sep1_)[core_mask_3898]
        self.theta_sep2 = np.array(theta_sep2_)[core_mask_3898]

        self.theta_uxp = np.array(theta_uxp_)
        self.theta_lxp = np.array(theta_lxp_)

        self.n0_pol_ped = np.array(n0_ped_)[core_mask_3898]
        self.n0_pol_sep1 = np.array(n0_sep1_)[core_mask_3898]
        self.n0_pol_sep2 = np.array(n0_sep2_)[core_mask_3898]

        self.S_at_ion_pol_sep1 = np.array(sna_sep1_)[core_mask_3898]
        self.S_at_ion_pol_sep2 = np.array(sna_sep2_)[core_mask_3898]

        self.pflux_pol_ped = np.array(n0_ped_)[core_mask_3898]
        self.pflux_pol_sep1 = np.array(pflux_sep1_)[core_mask_3898]
        self.pflux_pol_sep2 = np.array(pflux_sep2_)[core_mask_3898]
    

    def load_b2_reactions(self):
        
        fn = '{}/balance.nc'.format(self.fdir)
        ds = nc.Dataset(fn)
        
        vol = ds['vol']
        crx = ds['crx']
        cry = ds['cry']

        
        # particle balance

        sna_papl = ds['eirene_mc_papl_sna_bal']
        sna_pipl = ds['eirene_mc_pipl_sna_bal']
        sna_pmpl = ds['eirene_mc_pmpl_sna_bal']
        sna_pppl = ds['eirene_mc_pppl_sna_bal']

        sna_papl_ = (np.sum(sna_papl, axis=0)[1]/vol)[:,self.jxa]
        sna_pipl_ = (np.sum(sna_pipl, axis=0)[1]/vol)[:,self.jxa]
        sna_pmpl_ = (np.sum(sna_pmpl, axis=0)[1]/vol)[:,self.jxa]
        sna_pppl_ = (np.sum(sna_pppl, axis=0)[1]/vol)[:,self.jxa]

        sna_tot_ = sna_papl_ + sna_pipl_ + sna_pmpl_ + sna_pppl_

        self.papl = sna_papl_
        self.pipl = sna_pipl_
        self.pmpl = sna_pmpl_
        self.pppl = sna_pppl_
        self.sna_tot = sna_tot_

        
        # ion energy balance
        
        shi_eapl = ds['eirene_mc_eapl_shi_bal']
        shi_eipl = ds['eirene_mc_eipl_shi_bal']
        shi_empl = ds['eirene_mc_empl_shi_bal']
        shi_eppl = ds['eirene_mc_eppl_shi_bal']
        
        shi_eapl_ = (np.sum(shi_eapl, axis=0)/vol)[:,self.jxa]
        shi_eipl_ = (np.sum(shi_eipl, axis=0)/vol)[:,self.jxa]
        shi_empl_ = (np.sum(shi_empl, axis=0)/vol)[:,self.jxa]
        shi_eppl_ = (np.sum(shi_eppl, axis=0)/vol)[:,self.jxa]

        shi_tot_ = shi_eapl_ + shi_eipl_ + shi_empl_ + shi_eppl_
        
        self.eapl = shi_eapl_
        self.eipl = shi_eipl_
        self.empl = shi_empl_
        self.eppl = shi_eppl_
        self.shi_tot = shi_tot_


    def load_eirene_reactions(self):    
        
        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'

        # load in densities of all relevant species
        _, n_e_ = sut.B2pl("ne 0 0 sumz writ jxa f.y") # electron
        _, n_Dp_ = sut.B2pl("na 0 0 sumz writ jxa f.y") # D+ ion
        _, n_D2p_ = sut.B2pl("dib2 0 0 sumz writ jxa f.y") # D2+ ion
        _, n_D0_ = sut.B2pl("dab2 0 0 sumz writ jxa f.y") # D0 atom
        _, n_D20_ = sut.B2pl("dmb2 0 0 sumz writ jxa f.y") # D20 molecule

        # electron impact
        _, EI3_ = sut.B2pl("'AMJUEL' 'H.4' '2.1.5' 'EI' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # m^3s^-1 I think and I'm cute
        _, EI4_ = sut.B2pl("'AMJUEL' 'H.10' '2.1.5' 'EI' amju writ jxa f.y", wdir = self.fdir) # <sigma v Ep> 
        _, EI8_ = sut.B2pl("'AMJUEL' 'H.4' '2.2.9' 'EI' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # m^3s^-1
        _, EI14_ = sut.B2pl("'AMJUEL' 'H.4' '2.2.11' 'EI' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # I think this in m^3s^-1 already

        # dissociation
        _, DS9_ = sut.B2pl("'AMJUEL' 'H.4' '2.2.5g' 'DS' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # this one doesn't work
        _, DS10_ = sut.B2pl("'AMJUEL' 'H.4' '2.1.10' 'DS' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # this one doesn't work
        _, DS13_ = sut.B2pl("'AMJUEL' 'H.4' '2.2.12' 'DS' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # this is in cm^3s^-1
        _, DS15_ = sut.B2pl("'AMJUEL' 'H.4' '2.2.14' 'DS' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # this is in cm^3s^-1
        _, DS16_ = sut.B2pl("'AMJUEL' 'H.8' '2.2.14' 'DS' amju writ jxa f.y", wdir = self.fdir) # <sigma v Ep>

        # charge exchange
        _, CX5_ = sut.B2pl("'HYDHEL' 'H.1' '3.1.8' 'CX' amju writ jxa f.y", wdir = self.fdir) # sigma
        _, CX5b_ = sut.B2pl("'HYDHEL' 'H.3' '3.1.8' 'CX' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # this is in m^3s^-1 I think
        _, CX12_ = sut.B2pl("'AMJUEL' 'H.2' '3.2.3' 'CX' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # this looks like cm^3s^-1
        
        # elastic collisions
        _, EL11_ = sut.B2pl("'AMJUEL' 'H.0' '0.3T' 'EL' amju writ jxa f.y", wdir = self.fdir) # potential
        _, EL11b_ = sut.B2pl("'AMJUEL' 'H.1' '0.3T' 'EL' amju writ jxa f.y", wdir = self.fdir) # sigma
        _, EL11c_ = sut.B2pl("'AMJUEL' 'H.3' '0.3T' 'EL' amju writ jxa f.y", wdir = self.fdir) # <sigma v>
        
        # recombination
        _, RC17_ = sut.B2pl("'AMJUEL' 'H.4' '2.1.8' 'RC' amju writ jxa f.y", wdir = self.fdir) # <sigma v> # cm^3s^-1
        _, RC18_ = sut.B2pl("'AMJUEL' 'H.10' '2.1.8' 'RC' amju writ jxa f.y", wdir = self.fdir) # <sigma v Ep>

        os.chdir(olddir)

        # source of D+
        # some of these are in cm^-3 I think
        Sn_EI3 = np.array(n_e_)*np.array(n_D0_)*np.array(EI3_) # e + D ionization 
        Sn_EI14 = np.array(n_e_)*np.array(n_D2p_)*np.array(EI14_) # e + D2+ ionization
        Sn_DS13 = np.array(n_e_)*np.array(n_D2p_)*np.array(DS13_)/1e6 # e + D2+ dissociation
#       Sn_DS10 = np.array(n_e_)*np.array(n_D20_)*np.array(DS10_) # e + D2 dissociation
        Sn_CX12 = np.array(n_Dp_)*np.array(n_D20_)*np.array(CX12_)/1e6 # D+ + D2 charge exchange
        Sn_RC17 = np.array(n_e_)*np.array(n_Dp_)*np.array(RC17_)/1e12 # e + D+ recombination

        self.Sn_EI3 = Sn_EI3
        self.Sn_EI14 = Sn_EI14
        self.Sn_DS13 = Sn_DS13
        self.Sn_CX12 = Sn_CX12
        self.Sn_RC17 = Sn_RC17

        self.Sn_EI = Sn_EI3 + Sn_EI14
        self.Sn_DS = Sn_DS13
        self.Sn_CX = Sn_CX12 # does not constitute a net creation/sink
        self.Sn_RC = Sn_RC17

        self.Sn_m3s1 = self.Sn_EI + self.Sn_DS - self.Sn_RC

    def load_neutral_pressures(self, method='edens', return_plot=False):

        ## get neutral pressure throughout the vessel - will need to use fort46 file

        so = aurora.solps_case(b2fgmtry_path = '{}/b2fgmtry'.format(self.bdir),
                                b2fstate_path = '{}/b2fstate'.format(self.fdir))
   
        if method == 'edens':
   
            edena_eV_m3 = so.fort46['edena'][:,0]*1e6 # edena in cm3

            eV_m3_to_Pa = 1.602e-19
            Pa_to_Torr = 1/133
        
            edena_mTorr = edena_eV_m3*eV_m3_to_Pa*Pa_to_Torr*1e3
            p_mTorr = edena_mTorr*(2/3)

            ### calculate pressure at different parts of vessel ###
        
            # this is needed to undo the weird fort46 format
            tris = so.triangles.astype(int) # this will allow indexing of fort46 vals
            R_nodes = so.xnodes[tris]
            Z_nodes = so.ynodes[tris]
            p_mTorr_reshape = p_mTorr[tris]

            # coordinates for cryopump region
            R_cryo = [0.639, 0.701]
            Z_cryo = [0.5, 1e10]
            cryo_locs = (R_nodes > R_cryo[0]) & (R_nodes < R_cryo[1]) & (Z_nodes > Z_cryo[0]) & (Z_nodes < Z_cryo[1])
            self.p_mTorr_cryo = np.mean(list(set(p_mTorr_reshape[cryo_locs])))

            # coordinates for midplane gauge        
            R_omp = [0.904, 1e10]
            Z_omp = [-0.1, 0.1]
            omp_locs = (R_nodes > R_omp[0]) & (R_nodes < R_omp[1]) & (Z_nodes > Z_omp[0]) & (Z_nodes < Z_omp[1])
            self.p_mTorr_omp = np.mean(list(set(p_mTorr_reshape[omp_locs])))
            
            if return_plot:
                import matplotlib.pyplot as plt
                fig,ax = plt.subplots();
                ax.plot(so.xnodes, so.ynodes, '.')
                ax.plot(R_nodes[omp_locs], Z_nodes[omp_locs], 'o')
                plt.show()
        
        elif method == 'mflux':
        # use mass flux balance instead
            
            # this is as per M. Wigram's presentation about calculating the pressure using mass balance

            m_D = 3.34e-27
            T_D = 300 * 1.380649e-23 # room temperature in J


            # the way the indices work is as follows:
            # find what element number the segment is in DG, e.g. cryopump is element #78
            # in block 3b of input.dat, this will be the number on the right in the footer of the 
            # list of additional segments
            # the number on the left will be *one higher* than the index here, e.g. 75 : 78, is 
            # the cryopump segment, so index 74 should be used here
            
            seg_FCRYO = [74]
            seg_GSIDE = [41, 42, 43] #this is just the three middle segments
#            seg_GSIDE = list(np.arange(29, 55)) # this is all segments whose normals face radially on outer wall
            seg_EBOT = [22] # this is the segment that is below the opening - so should be more similar to omp
            seg_EBOT = [22, 71] # alternatively, it may make sense to take the average particle flux over this surface and the one at the bottom
            seg_BBOT = [71] # this is way at the bottom, so it should be its own thing

            #pflux_at_s1_cryo = so.fort44['wldna(0)'][cryo_seg_ind] # 0 is the sum over the strata
            #pflux_ml_s1_cryo = so.fort44['wldnm(0)'][cryo_seg_ind]
            #area_m2_cryo = so.fort44['wlarea'][cryo_seg_ind]           
 
            #pflux_s1m2_cryo = (1/2*pflux_at_s1_cryo + pflux_ml_s1_cryo) / area_m2_cryo

            #p_Pa_cryo = pflux_s1m2_cryo * np.sqrt(T_D * m_D) # in Pa
            #p_mTorr_cryo = p_Pa_cryo / 133 * 1000 # Pa --> Torr --> mTorr

            #self.p_mTorr_cryo = p_mTorr_cryo[0]

            pflux_s1m2 = np.zeros(4) # F-CRYO, G-SIDE, E-BOT, B-BOT

            pflux_s1m2[0] = calculate_pflux_for_pressure(so, seg_FCRYO)
            pflux_s1m2[1] = calculate_pflux_for_pressure(so, seg_GSIDE)
            pflux_s1m2[2] = calculate_pflux_for_pressure(so, seg_EBOT)
            pflux_s1m2[3] = calculate_pflux_for_pressure(so, seg_BBOT)

            p_Pa = pflux_s1m2 * np.sqrt(T_D * m_D) # in Pa
            p_mTorr = p_Pa / 133 * 1000 # Pa --> Torr --> mTorr

            self.p_mTorr_FCRYO, self.p_mTorr_GSIDE, self.p_mTorr_EBOT, self.p_mTorr_BBOT = p_mTorr

    
    def load_synth_diag(self):
        
        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'
    
        # load emissivity   
        _, emiss_ = sut.B2pl("eirc 0 1216 3 nspec 2.052e-17 rm* writ jxa f.y")
        self.emiss = np.array(emiss_)

        from pathlib import Path

        # try to load brightness if chords file exists
        chord_file_rev = 'LYMID_WALL_NEW_REV.chords'
        chord_path_rev = Path(self.fdir + '/' + chord_file_rev)
        if chord_path_rev.exists():
            print('File exists')
            _, bright_rev_ = sut.B2pl("writ eirc phys 0 1216 3 nspec 2.052e-17 rm* '{}' chor".format(chord_file_rev))
            while len(bright_rev_) < 20:
                bright_rev_.append(0)
            self.bright = np.array(bright_rev_)
        else:
            print('File not found')
            self.bright = np.zeros_like(self.emiss)        

        os.chdir(olddir)


    def load_transport(self):
        
        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'

        _, dn_ = sut.B2pl("d 0 0 sumz writ jxa f.y")
        _, vr_ = sut.B2pl("vlay 0 0 sumz writ jxa f.y")
        _, ke_ = sut.B2pl("kye 0 0 sumz writ jxa f.y")
        _, ki_ = sut.B2pl("kyi0 0 0 sumz writ jxa f.y")
        
        os.chdir(olddir)

        self.dn = np.array(dn_)
        self.vr = np.array(vr_)
        self.ke = np.array(ke_)
        self.ki = np.array(ki_)


    def load_fluxes(self):

        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'
    
        # particle flux
    
        _, pflux_ = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ {} f.y".format(self.jxa))
        _, diff_flux_ = sut.B2pl("fnay 1 zsel sy m/ writ {} f.y".format(self.jxa))
        _, conv_flux_ = sut.B2pl("na za m* vlay m* 0 0 sumz writ {} f.y".format(self.jxa))
        
        # poloidal flux as well
        _, pflux_theta_ = sut.B2pl("fnax za m* 0 0 sumz sx m/ writ {} f.y".format(self.jxa))


        # heat flux
        _, qe_ = sut.B2pl("fhey sy m/ writ {} f.y".format(self.jxa))
        _, qi_ = sut.B2pl("fhiy sy m/ writ {} f.y".format(self.jxa))
        
        os.chdir(olddir)

        self.pflux = np.array(pflux_)
        self.diff_flux = np.array(diff_flux_)
        self.conv_flux = np.array(conv_flux_)
        self.pflux_theta = np.array(pflux_theta_)
        self.qe = np.array(qe_)
        self.qi = np.array(qi_)
    
    def load_fsa(self, rad_cells=38):

        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'
    
        inner_lower_ind = 13
        inner_upper_ind = 35
        outer_upper_ind = 62
        outer_lower_ind = 84

        inner_inds = np.arange(inner_lower_ind, inner_upper_ind+1)
        outer_inds = np.arange(outer_upper_ind, outer_lower_ind+1)

        # this generates an array of only core indices
        core_mask_3898 = np.hstack((inner_inds, outer_inds))
        core_mask_3696 = core_mask_3898 - 1 # just need to shift the mask down 1 if using this notation

        # this will carry the totals
        pflux_tot = np.zeros(rad_cells)
        n0_tot = np.zeros(rad_cells)
        #S_ion_tot = np.zeros(rad_cells)

        for pol_ind in core_mask_3898:
            _, pflux_pol_ = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ {} f.y".format(pol_ind))
            _, n0_pol_ = sut.B2pl("dab2 0 0 sumz writ {} f.y".format(pol_ind))

            pflux_tot += np.array(pflux_pol_)
            n0_tot += np.array(n0_pol_)

        # compute averages
        pflux_fsa_ = pflux_tot / len(core_mask_3898)
        n0_fsa_ = n0_tot / len(core_mask_3898)
        #S_ion_fsa_ = S_ion_tot / len(core_mask_3898)

        self.pflux_fsa = pflux_fsa_
        self.n0_fsa = n0_fsa_

 
    def load_targets(self, lower = True):

        # get density, temperature, and Jsat? at the targets

        # decide whether to plot lower or upper targets
        if lower:
            inner_ind = 0
            outer_ind = 95
        else:
            inner_ind = 47
            outer_ind = 48

        # load density and temperature

        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'
       
        rho_IT_, ne_IT_ = sut.B2pl("ne 0 0 sumz writ {} f.y".format(inner_ind))
        rho_IT_, Te_IT_ = sut.B2pl("te 0 0 sumz writ {} f.y".format(inner_ind))
        rho_IT_, pflux_IT_ = sut.B2pl("fnax za m* 0 0 sumz sx m/ writ {} f.y".format(inner_ind))
        
        rho_OT_, ne_OT_ = sut.B2pl("ne 0 0 sumz writ {} f.y".format(outer_ind))
        rho_OT_, Te_OT_ = sut.B2pl("te 0 0 sumz writ {} f.y".format(outer_ind))
        rho_OT_, pflux_OT_ = sut.B2pl("fnax za m* 0 0 sumz sx m/ writ {} f.y".format(outer_ind))
        
        rho_T_map_, _ = sut.B2pl("ne 0 0 sumz writ {} f.y".format(self.jxa))

        os.chdir(olddir)

        self.rho_IT = np.array(rho_IT_)
        self.ne_IT = np.array(ne_IT_)
        self.Te_IT = np.array(Te_IT_)
        self.pflux_IT = np.array(pflux_IT_)

        self.rho_OT = np.array(rho_OT_)
        self.ne_OT = np.array(ne_OT_)
        self.Te_OT = np.array(Te_OT_)
        self.pflux_OT = np.array(pflux_OT_)
        
        self.rho_T_map = np.array(rho_T_map_)


    def check_conservation(self):

        # this function will compute the divergence of the flux and the source to compare against each other in two different ways
        
        ### pull in fluxes
        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'
    
        # flux density
        _, flux_R_density_ip1 = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ {} f.y".format(self.jxa+1))
        _, flux_R_density_i = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ {} f.y".format(self.jxa))
    
        _, flux_theta_density_jp1 = sut.B2pl("fnax za m* 0 0 sumz sx m/ writ {} f.y".format(self.jxa+1))
        _, flux_theta_density_j = sut.B2pl("fnax za m* 0 0 sumz sx m/ writ {} f.y".format(self.jxa))
        
        flux_R_density_ip1, flux_R_density_i = np.array(flux_R_density_ip1), np.array(flux_R_density_i)
        flux_theta_density_jp1, flux_theta_density_j = np.array(flux_theta_density_jp1), np.array(flux_theta_density_j)

        # fluxes
        _, flux_R_ip1 = sut.B2pl("fnay za m* 0 0 sumz writ {} f.y".format(self.jxa+1))
        _, flux_R_i = sut.B2pl("fnay za m* 0 0 sumz writ {} f.y".format(self.jxa))
        
        _, flux_theta_jp1 = sut.B2pl("fnax za m* 0 0 sumz writ {} f.y".format(self.jxa+1))
        _, flux_theta_j = sut.B2pl("fnax za m* 0 0 sumz writ {} f.y".format(self.jxa))

        flux_R_ip1, flux_R_i = np.array(flux_R_ip1), np.array(flux_R_i)
        flux_theta_jp1, flux_theta_j = np.array(flux_theta_jp1), np.array(flux_theta_j)
        
        os.chdir(olddir)

        ### pull in source
        fn = '{}/balance.nc'.format(self.fdir)
        ds = nc.Dataset(fn)

        sna = ds['eirene_mc_papl_sna_bal']
        vol = ds['vol']
        hx = ds['hx']
        hy = ds['hy']
    
        sna, vol, hx, hy = np.array(sna), np.array(vol), np.array(hx), np.array(hy)

        sna_sum = np.sum(sna,axis=0)
        sna_Dplus = sna_sum[1] # this is actually 0 in most places, so need to index +1
        
        sna_ij = sna_Dplus[:,self.jxa+1]
        sna_density_ij = sna_Dplus[:,self.jxa+1]/vol[:,self.jxa+1]
        
        # method 1 uses the flux density to compare against source density
        dGamma_dx_density = (flux_theta_density_jp1 - flux_theta_density_j)/(hx[:,self.jxa+1])
        dGamma_dy_density = (flux_R_density_ip1 - flux_R_density_i)/(hy[:,self.jxa+1])
        div_Gamma_density = dGamma_dx_density + dGamma_dy_density # this should be equal to sna_density_ij

        # method 2 uses the flux to compare against the source (as defined in solps)
        dGamma_dx = flux_theta_jp1 - flux_theta_j
        dGamma_dy = flux_R_ip1 - flux_R_i
        div_Gamma = dGamma_dx + dGamma_dy

        self.div_Gamma_density = div_Gamma_density
        self.div_Gamma = div_Gamma
        self.sna_density = sna_density_ij
        self.sna = sna_ij


def num_gradient(x, y):
    
    import math
    dydx = np.zeros_like(x)
    dydx[0] = math.nan
    dydx[1:] = np.diff(y)/np.diff(x)

    return dydx

def num_integral(x, y):

    import math
    I = np.zeros_like(y)
    cum_int = 0
    I[0] = math.nan
    
    for i in range(1, len(y)):

        if np.isnan(y[i]):
            I[i] = math.nan

        else:
            if np.isnan(y[i-1]):
                I[i] = 0
            else:
                cum_int += (y[i-1] + y[i])/2*(x[i] - x[i-1])
                I[i] = cum_int

    return I
            

def calculate_pflux_for_pressure(so, pump_inds):

    pflux_at_s1, pflux_ml_s1, area_m2 = 0, 0, 0
    
    for pind in pump_inds:

        pflux_at_s1 += so.fort44['wldna(0)'][pind]
        pflux_ml_s1 += so.fort44['wldna(0)'][pind]
        area_m2 += so.fort44['wlarea'][pind]

    pflux_s1m2 = (1/2*pflux_at_s1 + pflux_ml_s1) / area_m2

    return pflux_s1m2

