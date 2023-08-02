# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:09:05 2023

@author: user
"""

import subprocess

# subprocess.call(['ls'], shell=True) #using the call() function

def B2pl(cmds, debug=False):
    # import sys
    # import subprocess
    """
    runs B2plot with the commands used in the call and reads contents of the resulting
    b2plot.write file into two lists
    
    ** Make sure you've sourced the setup script first, or this won't work! **
    **  Make sure B2PLOT_DEV is set to 'ps'
    """

    if debug:
        cmdstr = 'echo "{}" | b2plot'.format(str(cmds))
        print(cmdstr)
    else:
        # cmdstr = 'echo "' + cmds + '" | b2plot'
        # cmdstr = 'echo "' + cmds + '" | b2plot >&/dev/null'
        cmdstr = 'echo "{}" | b2plot 2>/dev/null'.format(str(cmds))

    return cmdstr
        
a = B2pl("dab2 0 0 sumz writ jxa f.y", debug=False)
print(a)
subprocess.call(a, shell=True) #using the call() function

    
            




