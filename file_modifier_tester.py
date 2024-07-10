# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:44:02 2024

@author: ychuang
"""




simu_loc = 'C:/Users/ychuang/Documents/SOLPS_data/simulation_data/mast/027205'
simu_case = 'org_automatic_tool_testcase'
case_loc = '{}/{}'.format(simu_loc, simu_case)


b2mn_basicrunflag_dic = {'b2mndr_ntim': True, 'b2mndr_dtim': True, 'b2mndr_stim': True}

b2mn_basicrunvalue_dic = {'b2mndr_ntim': '1500', 'b2mndr_dtim': '6.0e-5', 'b2mndr_stim': '-1.0'}


b2mn_outputflag_dic = {'b2wdat_iout': True}

b2mn_outputvalue_dic = {'b2wdat_iout': '0'}


b2mn_flagdictpl_dic = {'basicrun': (b2mn_basicrunflag_dic, b2mn_basicrunvalue_dic) , 
                     'output': (b2mn_outputflag_dic, b2mn_outputvalue_dic)}



def flag_filter(flag_dic):

    mod_dic = {}
    
    for aa in list(flag_dic.keys()):
        
        if flag_dic[aa] == True:
            mod_dic[aa] = True
        
        elif flag_dic[aa] == False:
            pass
    
    return mod_dic
        
        

b2mn_loc = '{}/{}'.format(case_loc, 'b2mn.dat')
    


def b2mn_modifier():
    
    with open(b2mn_loc) as f:
         lines = f.readlines()
    
    for dickey in b2mn_flagdictpl_dic.keys():
        
        flagdictpl = b2mn_flagdictpl_dic[dickey]
        
        modb2mn_dic = flag_filter(flag_dic = flagdictpl[0])
        b2mnvalue_dic = flagdictpl[1]
        
        
        for j, string in enumerate(lines):
            
            for aa in list(modb2mn_dic.keys()):
                
                if aa in string:
                    print('{} is on line {}'.format(aa, str(j)))
                    
                    if dickey == 'basicrun':
                        
                        writelist = ''.join('\'{}\''.format(aa) + "\t\t\t" + "   " + '\'{}\''.format(b2mnvalue_dic[aa]))
                        lines[j] = writelist + "\n"
                    
                    elif dickey == 'output':
                        
                        writelist = ''.join('\'{}\''.format(aa) + "\t\t" + "       " + '\'{}\''.format(b2mnvalue_dic[aa]))
                        lines[j] = writelist + "\n"
                        
    
    
    m_gfile = '{}/prac_b2mn.dat'.format(case_loc)


    with open(m_gfile,'w') as g:
        for i, line in enumerate(lines):         ## STARTS THE NUMBERING FROM 1 (by default it begins with 0)    
            g.writelines(line)
    
    
        
        
b2mn_modifier()        
        
        
        
        