# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 18:37:40 2023

@author: user
"""

from os import path, environ
import numpy as np

def read_dsa(fname='dsa'):
    if not path.exists(fname):
        print('ERROR: b2fgmtry file not found:',fname)
        return None

    dsa = []
    with open(fname, 'r') as f:
        lines = f.readlines()

    for i,line in enumerate(lines):
        dsa.append(float(line.split()[0]))

    return dsa

def read_b2fgmtry(fname):
    if not path.exists(fname):
        print('ERROR: b2fgmtry file not found:',fname)
        return None

    data = []
    with open(fname, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):

        # Special handling for first few lines
        if i == 0:
            version = line.split()[0][7:-1]
            geo = {'version':version}
            continue
        elif i == 1:
            continue
        elif i == 2:
            # Assume starts with nx,ny after version
            geo['nx'] = int(line.split()[0])
            geo['ny'] = int(line.split()[1])
            numcells = (geo['nx']+2)*(geo['ny']+2)
            continue

        if line.split()[0] == '*cf:':
            vartype = line.split()[1]
            varsize = int(line.split()[2])
            varname = line.split()[3]
            # Some variables have no entries depending on config
            if varsize == 0:
                geo[varname] = None
            data = []
        else:
            # Parse by type
            if vartype == "char":
                geo[varname] = line.strip()
            else:
                splitline = line.split()
                for value in splitline:
                    if vartype == "int":
                        data.append(int(value))
                    else:
                        data.append(float(value))

                if len(data) == varsize:
                    if varsize%numcells == 0:
                        geo[varname] = np.array(data).reshape([geo['nx']+2,geo['ny']+2,int(varsize/numcells)], order = 'F')
                    else:
                        geo[varname] = np.array(data)

    return geo

def scrape_b2mn(fname = 'b2mn.dat'):
    b2mn = {}
    if not path.exists(fname):
        print('ERROR: b2mn.dat file not found:',fname)
        return b2mn
    
    with open(fname, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        sline = line.strip()
        if len(line) <= 1:
            continue
        if sline[0] == "#" or sline[0] == "*":
            continue
        else:
            count_quotes = 0
            quote_pos = []
            for i, c in enumerate(sline):
                if c == "'":
                    count_quotes = count_quotes + 1
                    quote_pos.append(i)
            # count_quotes = sline.count("'")
            if count_quotes == 2:
                # For cases where variable is enclosed in single quotes but value is not: 'b2mwti_jxa'
                thisvar = sline[quote_pos[0]+1:quote_pos[1]]
                this = sline[quote_pos[1]+1:-1]
            else:
                # Typically this is for 4 single quotes
                # For cases where both variable and value are enclosed in single quotes: 'b2mwti_jxa'   '36'
                #
                # This can occur when some comment after the value has quotes in it
                # e.g., "'b2sicf_phm0'  '0.0'  Old value '1.0'"
                thisvar = sline[quote_pos[0]+1:quote_pos[1]]
                this = sline[quote_pos[2]+1:quote_pos[3]]


            if thisvar == "b2mwti_jxa":
                b2mn['jxa'] = int(this)
            if thisvar == "b2tqna_inputfile":
                b2mn['b2tqna_inputfile'] = int(this)

    return b2mn

def read_b2fstate(fname):
    if not path.exists(fname):
        print('ERROR: b2fstate file not found: ',fname)
        return None

    DEBUG = False

    data = []
    with open(fname, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):

        # Special handling for first few lines
        if i == 0:
            version = line.split()[0][7:-1]
            state = {'version':version}
            continue
        elif i == 1:
            continue
        elif i == 2:
            # Assume starts with nx,ny,ns after version
            state['nx'] = int(line.split()[0])
            state['ny'] = int(line.split()[1])
            state['ns'] = int(line.split()[2])
            numcells = (state['nx']+2)*(state['ny']+2)
            continue

        if line.split()[0] == '*cf:':
            vartype = line.split()[1]
            varsize = int(line.split()[2])
            varname = line.split()[3]
            if DEBUG:
                print(varname,vartype,varsize,state['nx'],state['ny'],state['ns'],numcells)
            # Some variables have no entries depending on config
            if varsize == 0:
                state[varname] = None
            data = []
        else:
            # Parse by type
            if vartype == "char":
                state[varname] = line.strip()
            else:
                splitline = line.split()
                for value in splitline:
                    if vartype == "int":
                        data.append(int(value))
                    else:
                        data.append(float(value))

                if len(data) == varsize:
                    if varsize == numcells:
                        # This is a scalar quantity
                        state[varname] = np.array(data).reshape([state['nx']+2,state['ny']+2], order = 'F')
                    elif varsize == 2*numcells:
                        # This is a flux quantity
                        state[varname] = np.array(data).reshape([state['nx']+2,state['ny']+2,2], order = 'F')
                    elif varsize == numcells*state['ns']:
                        # This is a scalar quantity by species
                        state[varname] = np.array(data).reshape([state['nx']+2,state['ny']+2,state['ns']], order = 'F')
                    elif varsize == 2*numcells*state['ns']:
                        # This is a flux quantity by species
                        state[varname] = np.array(data).reshape([state['nx']+2,state['ny']+2,2,state['ns']], order = 'F')
                    elif varsize == 4*numcells:
                        # This is a flux quantity in 3.1 format
                        state[varname] = np.array(data).reshape([state['nx']+2,state['ny']+2,2,2], order = 'F')
                    elif varsize == 4*numcells*state['ns']:
                        # This is a flux quantity by species in 3.1 format
                        state[varname] = np.array(data).reshape([state['nx']+2,state['ny']+2,2,2,state['ns']], order = 'F')
                    elif varsize%numcells == 0:
                            print("Warning, must have missed some dimension checks for variable:",varname)
                    else:
                        # For other dimensions assign as is (e.g., zamin)
                        state[varname] = np.array(data)
    return state


def set_b2plot_dev(verbose = False):
    if 'B2PLOT_DEV' in environ.keys():
        if environ['B2PLOT_DEV'] == 'ps':
            if verbose:
                print('B2PLOT_DEV already set to ps')
        else:
            print("Changing environment variable B2PLOT_DEV to 'ps' for B2plot calls")
            environ['B2PLOT_DEV'] = 'ps'
    else:
        print('WARNING: Need to source setup.csh for a SOLPS-ITER distribution to enable B2plot calls')
        
def B2pl(cmds, wdir='.', debug=False):
    # import sys
    import subprocess
    """
    runs B2plot with the commands used in the call and reads contents of the resulting
    b2plot.write file into two lists
    
    ** Make sure you've sourced the setup script first, or this won't work! **
    **  Make sure B2PLOT_DEV is set to 'ps'
    """

    if debug:
        cmdstr = 'echo "' + cmds + '" | b2plot'
        print(cmdstr)
    else:
        # cmdstr = 'echo "' + cmds + '" | b2plot'
        # cmdstr = 'echo "' + cmds + '" | b2plot >&/dev/null'
        cmdstr = 'echo "' + cmds + '" | b2plot 2>/dev/null'
        testcmd = subprocess.check_output(cmdstr,shell=True)
        if testcmd == b'':
            print('\nERROR: b2plot command was not successful, is the case still running?')
            print('Command was: ',cmdstr)
            raise OSError
            
    fname = path.join(wdir, 'b2pl.exe.dir', 'b2plot.write')
    if not path.exists(fname):
        print('B2Plot writing failed for call:')
        print(cmds)
        print('in directory: ' + wdir + '\n')
        raise OSError

    # from IPython import embed; embed()

    x, y = [], []
    with open(fname) as f:
        lines = f.readlines()

    for line in lines:
        elements = line.split()
        if elements[0] == '#':
            pass
        else:
            x.append(float(elements[0]))
            y.append(float(elements[1]))
    x = x[0:(len(x) // 2)]  # used to be: x=x[0:(len(x)/2)-1], chopped final value
    y = y[0:(len(y) // 2)]

    return x, y        
        
        

def read_transport_files(fileloc, dsa=None, geo=None, state=None, force_read_inpufile=False):
    # Attempts to get transport coefficient data, including pinch term.
    # Fills using b2.transport.parameters, then b2.transport.inputfile if
    # b2mn.dat indicates inputfile is active. 
    #
    # Uses f90nml
    #
    # Inputs:
    # fileloc : location of b2mn.dat, b2.transport.*
    # dsa : dsa from read_dsa (or whatever dx_sep to interpolate onto)
    # geo : geometry dict from read_b2fgmtry 
    # state : state dict from read_b2fstate
    #
    # Note, geo and state only used to get dimensions, only really need
    # geo['ny'], state['ns']
    
    if dsa is None:
        dsa = read_dsa('dsa')
    
    if geo is None:
        geo = read_b2fgmtry('../baserun/b2fgmtry')

    if state is None:
        state = read_b2fstate('b2fstate')
        
    read_inputfile = False
    if force_read_inpufile:
        read_inputfile = True

    # Read b2mn.dat unless overridden by input argument
    if not read_inputfile:
        b2mn = scrape_b2mn("b2mn.dat")
        if ("b2tqna_inputfile" in b2mn.keys()):
            if b2mn["b2tqna_inputfile"] == 1:
                read_inputfile = True

    # Initialize variables we want defined. The final arrays will have
    # a dimension [ny+2] but b2.transport.inputfile may not!
    dn = np.zeros((geo['ny']+2,state['ns']))
    dp = np.zeros((geo['ny']+2,state['ns']))
    chii = np.zeros((geo['ny']+2,state['ns']))
    chie = np.zeros(geo['ny']+2)
    vlax = np.zeros((geo['ny']+2,state['ns']))
    vlay = np.zeros((geo['ny']+2,state['ns']))
    vsa = np.zeros((geo['ny']+2,state['ns']))
    sig = np.zeros(geo['ny']+2)
    alf = np.zeros(geo['ny']+2)    

    try:
        import f90nml
        parser = f90nml.Parser()
        parser.global_start_index=1
    except:
        print('f90nml required to read transport input files!')
        return None
    
    try:
        nml = parser.read('b2.transport.parameters')
        this = nml['transport']['parm_dna']
        for ispec in range(len(this)):
            dn[:,ispec] = this[ispec]
    except:
        pass
    
    if read_inputfile:
        try:
            nml = parser.read('b2.transport.inputfile')
            tdata = nml['transport']['tdata']
            nS = len(tdata)
            for ispec in range(nS):
                nKinds = len(tdata[ispec])
                for jkind in range(nKinds):
                    this = tdata[ispec][jkind]

                    # Check if this kind was filled with none by f90nml (not defined)
                    test = [i[1] for i in this]
                    if all(val is None for val in test):
                        continue
                    
                    # Read and interpolate back on dsa
                    xRaw = [i[0] for i in this] 
                    yRaw = [i[1] for i in this]
                    yInterp = np.interp(dsa,xRaw,yRaw)
                    
                    if jkind+1 == 1:
                        dn[:,ispec] = yInterp
                    elif jkind+1 == 2:
                        dp[:,ispec] = yInterp
                    elif jkind+1 == 3:
                        chii[:,ispec] = yInterp
                    elif jkind+1 == 4:
                        chie[:] = yInterp
                    elif jkind+1 == 5:
                        vlax[:,ispec] = yInterp                        
                    elif jkind+1 == 6:
                        vlay[:,ispec] = yInterp                        
                    elif jkind+1 == 7:
                        vsa[:,ispec] = yInterp                        
                    elif jkind+1 == 8:
                        sig[:] = yInterp                        
                    elif jkind+1 == 9:
                        alf[:] = yInterp                        
                    
        except:
            pass
        
        
    return dict(dn=dn, dp=dp, chii=chii, chie=chie, vlax=vlax, vlay=vlay, vsa=vsa, sig=sig, alf=alf)


def loadg(filename):
    infile = open(filename, 'r')
    lines = infile.readlines()

    # read first line for case string and grid size
    line = lines[0]
    words = line.split()

    nw = int(words[-2])
    nh = int(words[-1])
    psi = np.linspace(0, 1, nw)

    # read in scalar parameters
    #   note: word size of 16 characters each is assumed for all of the data to be read

    # line 1
    line = lines[1]
    rdim = float(line[0:16])
    zdim = float(line[16:32])
    rcentr = float(line[32:48])
    rleft = float(line[48:64])
    zmid = float(line[64:80])

    # line 2
    line = lines[2]
    rmaxis = float(line[0:16])
    zmaxis = float(line[16:32])
    simag = float(line[32:48])
    sibry = float(line[48:64])
    bcentr = float(line[64:80])

    # line 3
    line = lines[3]
    current = float(line[0:16])

    # read in profiles
    #   start by reading entire file into single list then split into individual profiles
    #   first block has 5 profiles of length nw and one array of length nh*nw

    temp = []
    count = 0
    lnum = 5
    terms = 5 * nw + nw * nh
    while count < terms:
        line = lines[lnum]
        numchar = len(line)
        nwords = numchar // 16
        count1 = 0
        while count1 < nwords:
            i1 = count1 * 16
            i2 = i1 + 16
            temp.append(float(line[i1:i2]))
            count1 += 1
            count += 1
        lnum += 1

    fpol = temp[0:nw]
    pres = temp[nw:2 * nw]
    ffprime = temp[2 * nw:3 * nw]
    pprime = temp[3 * nw:4 * nw]
    psirz_temp = temp[4 * nw:(4 + nh) * nw]
    qpsi = temp[(4 + nh) * nw:]

    # split psirz up into 2D matrix
    count = 0
    psirz = []
    while count < nh:
        ind1 = count * nw
        ind2 = ind1 + nw
        psirz.append(psirz_temp[ind1:ind2])
        count += 1

    # scalars for length of boundary and limiter arrays
    line = lines[lnum]
    words = line.split()
    nbbbs = int(words[0])
    limitr = int(words[1])

    # read boundary and limiter points into temp array

    temp = []
    count = 0
    terms = 2 * (nbbbs + limitr)
    lnum += 1
    while count < terms:
        line = lines[lnum]
        numchar = len(line)
        nwords = numchar // 16
        count1 = 0
        while count1 < nwords:
            i1 = count1 * 16
            i2 = i1 + 16
            temp.append(float(line[i1:i2]))
            count1 += 1
            count += 1
        lnum += 1
    bdry_temp = temp[0:(2 * nbbbs)]
    limit_temp = temp[(2 * nbbbs):]

    # split boundary into (R,Z) pairs
    count = 0
    rbdry = []
    zbdry = []
    while count < len(bdry_temp) - 1:
        rbdry.append(bdry_temp[count])
        zbdry.append(bdry_temp[count + 1])
        count += 2

    # split limiter into (R,Z) pairs
    count = 0
    rlim = []
    zlim = []
    while count < len(limit_temp) - 1:
        rlim.append(limit_temp[count])
        zlim.append(limit_temp[count + 1])
        count += 2

    g = dict(nw=nw, nh=nh, rdim=rdim, zdim=zdim, rcentr=rcentr, rleft=rleft, zmid=zmid,
             rmaxis=rmaxis, zmaxis=zmaxis, simag=simag, sibry=sibry, current=current,
             fpol=np.array(fpol),
             ffprime=np.array(ffprime), pprime=np.array(pprime), psirz=np.array(psirz),
             qpsi=np.array(qpsi), nbbbs=nbbbs, bcentr=bcentr,
             pres=np.array(pres), limitr=limitr, rbdry=np.array(rbdry),
             zbdry=np.array(zbdry), rlim=np.array(rlim), zlim=np.array(zlim))

    return g
