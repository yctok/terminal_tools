# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:32:34 2024

@author: user
"""
import SOLPSutils as sut
import os



def load_eirene_reactions(self):    
    
    working = str(self.fdir)
    olddir = os.getcwd()
    os.chdir(working)
    os.environ['B2PLOT_DEV'] = 'ps'

    # load in densities of all relevant species
    _, n_e_ = sut.B2pl("ne 0 0 sumz writ jxa f.y") # electron
    
    