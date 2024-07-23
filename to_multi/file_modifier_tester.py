# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:44:02 2024

@author: ychuang
"""

import test_dir as td
import os

# simu_loc, simu_case, case_loc = td.test_dir()

# simu_loc = 'C:/Users/ychuang/Documents/SOLPS_data/simulation_data/mast/027205'
# simu_case = 'org_automatic_tool_testcase'
# case_loc = '{}/{}'.format(simu_loc, simu_case)


def flag_filter(flag_dic):

    mod_dic = {}
    
    for aa in list(flag_dic.keys()):
        
        if flag_dic[aa] == True:
            mod_dic[aa] = True
        
        elif flag_dic[aa] == False:
            pass
    
    return mod_dic
        
        

# b2mn_loc = '{}/{}'.format(case_loc, 'b2mn.dat')
    


def b2mn_modifier(b2mn_loc, case_loc, run_type):
    
    with open(b2mn_loc) as f:
         lines = f.readlines()
    
    b2mn_flagdictpl_dic = td.b2mn_set(run_type)
    
    
    for dickey in b2mn_flagdictpl_dic.keys():
        
        flagdictpl = b2mn_flagdictpl_dic[dickey]
        
        modb2mn_dic = flag_filter(flag_dic = flagdictpl[0])
        b2mnvalue_dic = flagdictpl[1]
        
        
        for j, string in enumerate(lines):
            
            for aa in list(modb2mn_dic.keys()):
                
                if aa in string:
                    # print('{} is on line {}'.format(aa, str(j)))
                    
                    if dickey == 'basicrun':
                        
                        writelist = ''.join('\'{}\''.format(aa) + "\t\t\t" + "   " + '\'{}\''.format(b2mnvalue_dic[aa]))
                        lines[j] = writelist + "\n"
                        print('{} is currently {}'.format(aa, b2mnvalue_dic[aa]))
                    
                    elif dickey == 'output':
                        
                        writelist = ''.join('\'{}\''.format(aa) + "\t\t" + "       " + '\'{}\''.format(b2mnvalue_dic[aa]))
                        lines[j] = writelist + "\n"
                        
                        print('{} is currently {}'.format(aa, b2mnvalue_dic[aa]))
                        
    
    
    # m_gfile = '{}/prac_b2mn.dat'.format(case_loc)


    with open(b2mn_loc,'w') as g:
        for i, line in enumerate(lines):         ## STARTS THE NUMBERING FROM 1 (by default it begins with 0)    
            g.writelines(line)
    
    # f_dir = os.getcwd()
    
    print('new b2mn is created in {}'.format(case_loc))
        
        
# b2mn_modifier()        


def batch_modifier(batch_loc, fname, case_loc, idchange):
    
    with open(batch_loc) as f:
         lines = f.readlines()
    
    key_list = ['#PBS -N', 'Initiated', 'cd', 'completed']
    
    "idchange list"
    
    dev = 'MAST'
    account_id = 'ychuang'
    email = 'ychuang@wm.edu'
    
    
    index_dic = {}
    
    for j, string in enumerate(lines):
        
        for aa in key_list:
                      
            if aa in string:
                print('{} is on line {}'.format(aa, str(j)))
                index_dic[aa] = j
    
    print(index_dic)
    
    k1 = index_dic['#PBS -N']
    
    list3 = lines[k1].split()
    
    if idchange == False:
        list3[2] = '{}_MAST_ychuang'.format(fname)
    
    elif idchange == True:
        
        list3[2] = '{}_{}_{}'.format(fname, dev, account_id)
        
    
    
    
    writelist = ''.join(x + ' ' for x in list3)
    lines[k1] = writelist + "\n"
    print('{} is currently {}'.format('#PBS -N', fname))
    

    k2 = index_dic['Initiated']

    list1 = lines[k2].split()
    list1[2] = fname
    
    if idchange == False:
        pass
    elif idchange == True:
        list1[-1] = email
        list1[1] = '\"{}'.format(dev)
    
    print(list1)
    
    
    writelist = ''.join(x + ' ' for x in list1)
    lines[k2] = writelist + "\n"
    print('{} is currently {}'.format('Initiated', fname))
           
    k3 = index_dic['cd']
    
    writelist = ''.join('cd' + ' ' + case_loc)
    lines[k3] = writelist + "\n"
    print('{} is currently {}'.format('cd', fname))
    
    k4 = index_dic['completed']

    list2 = lines[k4].split()
    list2[2] = fname
    
    if idchange == False:
        pass
    elif idchange == True:
        list2[-1] = email
        list2[1] = '\"{}'.format(dev)
    
    
    
    
    writelist = ''.join(x + ' ' for x in list2)
    lines[k4] = writelist + "\n"
    print('{} is currently {}'.format('completed', fname))
    
    with open(batch_loc,'w') as g:
        for i, line in enumerate(lines):         ## STARTS THE NUMBERING FROM 1 (by default it begins with 0)    
            g.writelines(line)
    
    # f_dir = os.getcwd()
    
    print('new batch is created in {}'.format(case_loc))




def b2boundary_modifier(b2mn_loc, case_loc, run_type):
    
    with open(b2mn_loc) as f:
         lines = f.readlines()
    
    b2mn_flagdictpl_dic = td.b2mn_set(run_type)
    
    
    for dickey in b2mn_flagdictpl_dic.keys():
        
        flagdictpl = b2mn_flagdictpl_dic[dickey]
        
        modb2mn_dic = flag_filter(flag_dic = flagdictpl[0])
        b2mnvalue_dic = flagdictpl[1]
        
        
        for j, string in enumerate(lines):
            
            for aa in list(modb2mn_dic.keys()):
                
                if aa in string:
                    # print('{} is on line {}'.format(aa, str(j)))
                    
                    if dickey == 'basicrun':
                        
                        writelist = ''.join('\'{}\''.format(aa) + "\t\t\t" + "   " + '\'{}\''.format(b2mnvalue_dic[aa]))
                        lines[j] = writelist + "\n"
                        print('{} is currently {}'.format(aa, b2mnvalue_dic[aa]))
                    
                    elif dickey == 'output':
                        
                        writelist = ''.join('\'{}\''.format(aa) + "\t\t" + "       " + '\'{}\''.format(b2mnvalue_dic[aa]))
                        lines[j] = writelist + "\n"
                        
                        print('{} is currently {}'.format(aa, b2mnvalue_dic[aa]))
                        
    
    
    # m_gfile = '{}/prac_b2mn.dat'.format(case_loc)


    with open(b2mn_loc,'w') as g:
        for i, line in enumerate(lines):         ## STARTS THE NUMBERING FROM 1 (by default it begins with 0)    
            g.writelines(line)
    
    # f_dir = os.getcwd()
    
    print('new b2mn is created in {}'.format(case_loc))
    
    
    
        
        
        