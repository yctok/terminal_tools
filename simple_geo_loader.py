# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:30:55 2024

@author: ychuang
"""

import numpy as np
import SOLPS_set as sps
from SOLPS_load_directory import load_directory 
import load_mast_expdata_method as lmem
import load_B2_data_method as lbdm
import load_coord_method as lcm
import fitting_method as fm 
from scipy import interpolate
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt



simu_loc = 'C:/Users/ychuang/Documents/SOLPS_data/simulation_data/mast/027205'
simu_case = 'mg_d3d'
file_dir = '{}/{}/baserun'.format(simu_loc, simu_case)
geo_dir = '{}/b2fgmtry'.format(file_dir)


simu_loc = 'C:/Users/ychuang/Documents/SOLPS_data/simulation_data/mast/027205'
simu_case_2 = 'org_new_series'
file_dir_2 = '{}/{}/baserun'.format(simu_loc, simu_case_2)
geo_dir_2 = '{}/b2fgmtry'.format(file_dir_2)



def load_b2fgmtry(file_name):
          
    try:
        geo = lcm.read_b2fgmtry(file_name)
        # print(type(geo))
    except:
        print('can not generate geo')
    
    return geo
    

geo_dic = {}

geo_dic['dot3'] = load_b2fgmtry(file_name= geo_dir)
geo_dic['MAST'] = load_b2fgmtry(file_name= geo_dir_2)

topic = 'mag_pol'


def mag_geo_pol(itername, pol_list, data, art_text,
                      axs, color_dic, psi_dic, A_dic, no_A_label):
        
        sk = int(pol_list[0])
        sd = int(pol_list[-1]) + 1
            
        
        plot_dat = data[sk:sd, psi_dic['st']:psi_dic['ed'], ]
        
        axs.add_artist(art_text)
        
        if no_A_label:
            
            axs.plot(pol_list, plot_dat[:, 0], linestyle='-', 
                color= color_dic[itername])
        else:
            
            axs.plot(pol_list, plot_dat[:, 0], linestyle='-', 
                color= color_dic[itername], label= 'A = {}'.format(A_dic[itername]))



if topic == 'mag_pol':
    

    
    psi_st = 19
    psi_ed = 38
    
    psi_dic = {'st': psi_st, 'ed': psi_ed}
    
    
    pol_list_a = []
    for i in range(36):
        pol_list_a.append('{}'.format(28 + i))
          
    
    
    flux_list = ['bx', 'B', 'bz']
    
    
    fig, axs = plt.subplots(3, 1)
    
    color_dic = {'MAST': 'red','dot3': 'purple'}
    A_dic = {'MAST': '1.4', 'dot3': '2.0'}
    
    B_text = AnchoredText('{}'.format('B: [T]'), 
                                 loc='upper center')
    
    bx_text = AnchoredText('{}'.format('$B_x$: [T]'), 
                                 loc='upper center')
    
    bz_text = AnchoredText('{}'.format('$B_z$: [T]'), 
                                 loc='lower center')
    
    text_list = [bx_text, B_text, bz_text]
    case_list = ['MAST', 'dot3']
    
    for ind, dat_name in enumerate(flux_list):
        
        for aa in case_list:
            
            mag_dat = geo_dic[aa]['bb'][:, :, -ind]
            
            mag_geo_pol(itername = aa, pol_list = pol_list_a, 
                data = mag_dat, art_text = text_list[ind], axs = axs[ind], 
                color_dic = color_dic, psi_dic = psi_dic, A_dic = A_dic, no_A_label = False)
                
        
        axs[ind].legend(loc= 'upper right')
    
    axs[2].set_xlabel('poloidal index')
    axs[0].set_title('magnetic strength at the separatrix')
    
    
    plt.subplots_adjust(hspace=.0)


fig, axs = plt.subplots()

color_dic = {'MAST': 'red','dot3': 'purple'}
A_dic = {'MAST': '1.4', 'dot3': '2.0'}

B_text = AnchoredText('{}'.format('B: [T]'), 
                             loc='upper center')

bx_text = AnchoredText('{}'.format('$B_x$: [T]'), 
                             loc='upper center')

bz_text = AnchoredText('{}'.format('$B_z$: [T]'), 
                             loc='lower center')

text_list = [B_text, bx_text, bz_text]
case_list = ['MAST', 'dot3']

    
for aa in case_list:
    
    mag_dat = geo_dic[aa]['bb'][:, :, -2]
    
    mag_geo_pol(itername = aa, pol_list = pol_list_a, 
        data = mag_dat, art_text = text_list[ind], axs = axs, 
        color_dic = color_dic, psi_dic = psi_dic, A_dic = A_dic, no_A_label = False)
            
    
    axs.legend(loc= 'upper right')

axs.set_xlabel('poloidal index')
axs.set_title('magnetic strength at the separatrix')


