# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:50:06 2024

@author: ychuang
"""



def tester_dir():
    
    simu_loc = 'C:/Users/ychuang/Documents/SOLPS_data/simulation_data/mast/027205'
    simu_case = 'org_automatic_tool_testcase'
    case_loc = '{}/{}'.format(simu_loc, simu_case)
    
    return simu_loc, simu_case, case_loc
    

def b2mn_set(run_type):

    b2mn_basicrunflag_dic = {'b2mndr_ntim': True, 'b2mndr_dtim': True, 'b2mndr_stim': True}
    if run_type == 'longrun':
        
        b2mn_basicrunvalue_dic = {'b2mndr_ntim': '1500', 'b2mndr_dtim': '6.0e-5', 'b2mndr_stim': '-1.0'}
    elif run_type == 'ioutrun':
        
        b2mn_basicrunvalue_dic = {'b2mndr_ntim': '5', 'b2mndr_dtim': '6.0e-5', 'b2mndr_stim': '-1.0'}
    elif run_type == 'testrun':
        
        b2mn_basicrunvalue_dic = {'b2mndr_ntim': '200', 'b2mndr_dtim': '6.0e-5', 'b2mndr_stim': '-1.0'}
    
    
    b2mn_outputflag_dic = {'b2wdat_iout': True}
    if run_type == 'ioutrun':
        
        b2mn_outputvalue_dic = {'b2wdat_iout': '4'}
    
    elif run_type == 'longrun' or run_type == 'testrun':
        
        b2mn_outputvalue_dic = {'b2wdat_iout': '0'}
    
    

    b2mn_flagdictpl_dic = {'basicrun': (b2mn_basicrunflag_dic, b2mn_basicrunvalue_dic) , 
                         'output': (b2mn_outputflag_dic, b2mn_outputvalue_dic)}
    
    return b2mn_flagdictpl_dic

    
    