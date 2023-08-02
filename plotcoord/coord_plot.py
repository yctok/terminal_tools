# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 14:09:19 2023

@author: user
"""

import os
import numpy as np
import coord_sutils as cs
from scipy import interpolate



class SOLPSplot:

    def __init__(self, workdir, gfile_loc):
        """
        Inputs:
          workdir         Directory with the SOLPS outputs
          gfile_loc       location of corresponding g file
          impurity_list   List of all the impurity species included in the plasma simulation
        """
        
        
        self.b2plot_ready = False
        if 'B2PLOT_DEV' in os.environ.keys():
            if os.environ['B2PLOT_DEV'] == 'ps':
                self.b2plot_ready = True

        self.data = {'workdir':workdir, 'gfile_loc': gfile_loc,
                     'expData':{'fitProfs':{}}, 'solpsData':{}}




    def calcPsiVals(self, plotit = False, dsa = None, b2mn = None, geo = None, verbose=True, shift=0):
        """
        Call b2plot to get the locations of each grid cell in psin space
    
        Saves the values to dictionaries in self.data['solpsData']
    
        Find grid corners first:
          0: lower left
          1: lower right
          2: upper left
          3: upper right
    
        Average location of cells 0 and 2 for middle of 'top' surface, 
        which is the top looking at outboard midplane
        Don't average over whole cell, dR << dZ at outboard midplane 
        and surface has curvature, so psin will be low
    
        jxa = poloidal cell index for the outer midplane
        crx = radial coordinate corner of grid [m]
        cry = vertical coordinate corner of grid [m]
        writ = write b2plot.write file
        f.y = plot against y
        """
    
        wdir = self.data['workdir']
    
        try:
            if dsa is None:
                dsa = cs.read_dsa('dsa')
            if geo is None:
                geo = cs.read_b2fgmtry('../baserun/b2fgmtry')
            if b2mn is None:
                b2mn = cs.scrape_b2mn("b2mn.dat")                
    
            crLowerLeft = geo['crx'][b2mn['jxa']+1,:,0]
            crUpperLeft = geo['crx'][b2mn['jxa']+1,:,2]
            czLowerLeft = geo['cry'][b2mn['jxa']+1,:,0]
            czUpperLeft = geo['cry'][b2mn['jxa']+1,:,2]               
        except:
            if verbose:
                print('  Failed to read geometry files directly, trying b2plot')
            if not self.b2plot_ready:
                cs.set_b2plot_dev(verbose=verbose)
                self.b2plot_ready = True
    
            try:
                dsa, crLowerLeft = cs.B2pl('0 crx writ jxa f.y', wdir = wdir)
            except Exception as err:
                print('Exiting from calcPsiVals')
                raise err
        
            # Only 2 unique psi values per cell, grab 0 and 2
            dummy, crUpperLeft = cs.B2pl('2 crx writ jxa f.y', wdir = wdir)  # all x inds are the same
            dummy, czLowerLeft = cs.B2pl('0 cry writ jxa f.y', wdir = wdir)
            dummy, czUpperLeft = cs.B2pl('2 cry writ jxa f.y', wdir = wdir)
        
        print(type(czUpperLeft))    
        
        ncells = len(czLowerLeft)
    
        g = cs.loadg(self.data['gfile_loc'])
        d = float(shift)
        psiN = (g['psirz'] - g['simag']) / (g['sibry'] - g['simag'])
    
        dR = g['rdim'] / (g['nw'] - 1)
        dZ = g['zdim'] / (g['nh'] - 1)
    
        gR = []
        for i in range(g['nw']):
            gR.append(g['rleft'] + i * dR + d)
    
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
            
        psi_list = psi_solps.tolist()
        RLL_list = crLowerLeft.tolist()
        ZLL_list = czLowerLeft.tolist()
        RUL_list = crUpperLeft.tolist()
        ZUL_list = czUpperLeft.tolist()
        
        print(type(ZUL_list))
        print(type(dsa))
        print(type(psi_list))
        
    
        self.data['solpsData']['crLowerLeft'] = RLL_list
        self.data['solpsData']['czLowerLeft'] = ZLL_list
        self.data['solpsData']['crUpperLeft'] = RUL_list
        self.data['solpsData']['czUpperLeft'] = ZUL_list
        self.data['solpsData']['dsa'] = dsa
        self.data['solpsData']['psiSOLPS'] = psi_list
        
        datakey = ['crLowerLeft','czLowerLeft', 'crUpperLeft','czUpperLeft', 
                   'dsa', 'psiSOLPS']
        cn = len(datakey)
        print(len(RLL_list))
        print(len(ZLL_list))
        print(len(RUL_list))
        print(len(ZUL_list))
        print(len(dsa))
        print(len(psi_list))
        
        dataindex = [RLL_list, ZLL_list, RUL_list, ZUL_list, dsa, psi_list]
        
        
        with open('../../../../../../repository/plotcoord/coord.txt', 'w') as file:
            colcount = 0    # Track the column number

            for x in datakey:
                # First thing we do is add one to the column count when
                # starting the loop. Since we're doing some math on it below
                # we want to make sure we don't divide by zero.
                colcount += 1
            
                # After each entry, add a tab character ("\t")
                file.write(x + "\t\t\t")
            
                # Now, check the column count against the MAX_COLUMNS. We
                # use a modulus operator (%) to get the remainder after dividing;
                # any number divisible by 3 in our example will return '0'
                # via modulus.
                if colcount == cn:
                    # Now write out a new-line ("\n") to move to the next line.
                    file.write("\n")

            for i in range(len(dsa)):
                for p in dataindex:
                    file.write(str(p[i]) + "\t\t\t")
                file.write("\n")

            
            # file.write('\n'.join(str(i) for i in dsa))
        
        
        
        
        
        