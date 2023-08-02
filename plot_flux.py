# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:15:51 2023

@author: user
"""

import SOLPSutils as sut
import matplotlib.pyplot as plt

xa, f0 = sut.B2pl("fnay za m* 0 0 sumz sy m/ writ jxa f.y")
xb, qe0 = sut.B2pl("fhey sy m/ writ jxa f.y")
xc, qi0 = sut.B2pl("fhiy sy m/ writ jxa f.y")
xd, ptheta = sut.B2pl("fnax za m* 0 0 sumz sx m/ writ jxa f.y")

xe, n0 = sut.B2pl("dab2 0 0 sumz writ jxa f.y")
xf, T0 = sut.B2pl("tab2 0 0 sumz writ jxa f.y")


plt.figure(1)
plt.scatter(xa, f0, label= 'particle flux')
plt.xlabel('Radial coordinate: $R- R_{sep}$', fontdict={"family":"Calibri","size": 20})
plt.ylabel('particle flux', fontdict={"family":"Calibri","size": 20})
plt.title('particle flux',fontdict={"family":"Calibri","size": 20})
plt.legend()

plt.figure(2)
plt.scatter(xb, qe0, label= 'electron heat flux')
plt.xlabel('Radial coordinate: $R- R_{sep}$', fontdict={"family":"Calibri","size": 20})
plt.ylabel('electron heat flux', fontdict={"family":"Calibri","size": 20})
plt.title('electron heat flux',fontdict={"family":"Calibri","size": 20})
plt.legend()

plt.figure(3)
plt.scatter(xc, qi0, label= 'ion heat flux')
plt.xlabel('Radial coordinate: $R- R_{sep}$', fontdict={"family":"Calibri","size": 20})
plt.ylabel('ion heat flux', fontdict={"family":"Calibri","size": 20})
plt.title('ion heat flux',fontdict={"family":"Calibri","size": 20})
plt.legend()

plt.figure(4)
plt.scatter(xd, ptheta, label= 'poloidal flux')
plt.xlabel('Radial coordinate: $R- R_{sep}$', fontdict={"family":"Calibri","size": 20})
plt.ylabel('poloidal flux', fontdict={"family":"Calibri","size": 20})
plt.title('poloidal flux',fontdict={"family":"Calibri","size": 20})
plt.legend()

plt.figure(5)
plt.scatter(xe, n0, label= 'neutral density')
plt.xlabel('Radial coordinate: $R- R_{sep}$', fontdict={"family":"Calibri","size": 20})
plt.ylabel('neutral density', fontdict={"family":"Calibri","size": 20})
plt.title('neutral density',fontdict={"family":"Calibri","size": 20})
plt.legend()

plt.figure(6)
plt.scatter(xf, T0, label= 'neutral temperature')
plt.xlabel('Radial coordinate: $R- R_{sep}$', fontdict={"family":"Calibri","size": 20})
plt.ylabel('neutral temperature', fontdict={"family":"Calibri","size": 20})
plt.title('neutral temperature',fontdict={"family":"Calibri","size": 20})
plt.legend()



plt.show()
