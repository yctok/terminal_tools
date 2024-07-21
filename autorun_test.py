# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:09:42 2024

@author: ychuang
"""

#Import os module
import os
import file_modifier_tester as fmt

sim_dir = os.getcwd()

#Print current working directory
print("Current directory:" , os.getcwd())

file_list = os.listdir(sim_dir)

print(file_list)



"""
envir_name = os.environ['OS']
print(envir_name)

"""


def sep_plot(case_loc, sep_type):
    
    os.chdir(case_loc)
    
    if sep_type == 'ne':
        os.system('2dt nesepm &')
    
    elif sep_type == 'te':
        
        os.system('2dt tesepm &')
        

def last10_plot(case_loc, name_last10):
    
    os.chdir(case_loc)
    os.system('2d_profiles')
    os.system('xyplot < {}.last10 &'.format(name_last10))
    
    
    

def multi_plot(sim_dir, file_list, scenario_type):
    
    for fname in file_list:
        
        if fname != 'baserun':
            
            case_loc = '{}/{}'.format(sim_dir, fname)
            # b2mn_loc = '{}/{}'.format(case_loc, 'b2mn.dat')
            
            if scenario_type == 'plot_sep':
                
                sep_plot(case_loc = case_loc, sep_type = 'ne')
            
            elif scenario_type == 'plot_last10':
                
                last10_plot(case_loc = case_loc, name_last10 = 'ne3da')
            
            
            elif scenario_type == 'runxport':
                
                os.chdir(case_loc)
                
                os.system('python3 ../../../../SOLPSxport/SOLPSxport_dr.py -g ../../gnpfiles/g027205.00275 -p ../../gnpfiles/fit_027205_275.dat -t 275 -f 0.5 -sh 0')
                
            
            elif scenario_type == 'no':
                pass



def multifile_creater(gen_folder):
    
    tk = 1
    tm = 4.0
    
    for tt in range(10):
        
        
        new_folder = '{}_n{:.1f}_tttest'.format(tk, tm)
        
        os.system('cp -r {} {}'.format(gen_folder, new_folder))
        
        tk = tk + 1
        tm = tm + 0.5
    
    




def multi_run(sim_dir, file_list, scenario_type):
    
    for fname in file_list:
        
        if fname != 'baserun':
            
            case_loc = '{}/{}'.format(sim_dir, fname)
            b2mn_loc = '{}/{}'.format(case_loc, 'b2mn.dat')
            batch_loc = '{}/{}'.format(case_loc, 'batch_example')
            
            if scenario_type == 'ioutrun':
                
                os.chdir(case_loc)
                
                fmt.b2mn_modifier(b2mn_loc = b2mn_loc, case_loc = case_loc, 
                                  run_type = 'ioutrun')
                
                
                os.system('cp b2fstate b2fstati')
                os.system('rm b2mn.prt')
                os.system('qsub batch_example')
            
            elif scenario_type == 'create file':
                
                gen = 'test_folder'
                
                multifile_creater(gen_folder = gen)
                
            elif scenario_type == 'change batch':
                
                tk = fname.split('_')[-1]
                
                if tk == 'tttest':
                    
                    fmt.batch_modifier(batch_loc = batch_loc, fname = fname, 
                                   case_loc = case_loc, idchange = True)
                
            
            elif scenario_type == 'more to discover':
                pass

test = 'run'

if test == 'plot':
    
    multi_plot(sim_dir = sim_dir, file_list = file_list, scenario_type = 'plot_last10')
    
elif test == 'run':
    
    multi_run(sim_dir = sim_dir, file_list = file_list, scenario_type = 'change batch')
    







"""

#Create a new directory
os.mkdir("test_folder")

#Change current working directory
os.chdir("test_folder")

#Print current working directory
print("Current directory now:" , os.getcwd())


os.system('cp b2fstate b2fstati')
os.system('rm b2mn.prt')

"""





