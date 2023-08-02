# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:46:15 2023

@author: user
"""

import os
import sys
import json
import numpy as np
import SOLPSutils as sut


workdir=os.getcwd()
print(workdir)

dsa = sut.read_dsa("dsa")

print(type(dsa))
with open('../../../../../SOLPSxport/dsa.txt', 'w') as file:
     file.write(json.dumps(dsa))


b2mn = sut.scrape_b2mn("b2mn.dat")

with open('../../../../../SOLPSxport/b2mn.txt', 'w') as nf:
     nf.write(json.dumps(b2mn))