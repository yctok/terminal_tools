# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:50:59 2023

@author: user
"""

import os
import SOLPSutils as sut
import matplotlib.pyplot as plt

class SOLPSplot:

    def __init__(self, dev, shift, series, case, jxa):

        self.dev = dev # this should be the shot used to generate the grid, so correct geqdsk is found
        self.shift = shift
        self.series = series
        self.case = case
        self.jxa = jxa
        self.SOLPSWORK = '/sciclone/data10/ychuang/solps-iter/runs'
        self.specific = '/' + str(self.dev) + '/' + str(self.shift) + '/' + str(self.series) + '/' + str(self.case) + '/'
        self.fdir = self.SOLPSWORK + str(self.specific)

    def neutral_plot(self):

        working = str(self.fdir)
        olddir = os.getcwd()
        os.chdir(working)
        os.environ['B2PLOT_DEV'] = 'ps'
        
        ## get neutral density and source at the midplane
        
        n0, x = sut.B2pl("dab2 0 0 sumz writ {} f.y".format(self.jxa))
        T0, x = sut.B2pl("tab2 0 0 sumz writ {} f.y".format(self.jxa))
        
        from IPython import embed; embed()
        
        os.chdir(olddir)
        
        return n0, T0
        
instance = SOLPSplot(dev = 'mast', shift = 'org_027205', series = 'p_ts4_series', case= '7ex_p4_d6_9_ts6_a', jxa = '57')
b = instance.neutral_plot()
# print(b)