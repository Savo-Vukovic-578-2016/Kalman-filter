# -*- coding: utf-8 -*-

#Kalman_filter 
#@author Savo Vukovic"

#Simple Kalman filther


import matplotlib.pyplot as plt
import numpy as np
import random

plt.rcParams['figure.figsize'] = (10, 8)

# Pocetni parametri
n = 50 # broj iteracija
sz = (n,) # size of array
x = -0.37727 # prava vrednost
#t = np.linspace(0, 2*np.pi, n)
#x = np.sin(t)
z = np.random.normal(x,0.1,size=sz) # merenja
Q = 1e-5 # sum


xhat=np.zeros(sz)      # Estimates for x, procene za x tj x^s
P=np.zeros(sz)         # Estimate error, greska u proceni
xhatminus=np.zeros(sz) # prior estimate, prethodna procena
Pminus=np.zeros(sz)    # prior error estimate, prethodna greska
KG=np.zeros(sz)         # Kalman gain
    
R = 0.1**2 # neka greska u merenju

# pocetne pretpostavke
xhat[0] = 0.0
P[0] = 1.0




for k in range(1,n):
    # update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+ Q

    # merenja
    KG[k] = Pminus[k]/( Pminus[k]+R )
    
    
    
    xhat[k] = xhatminus[k]+KG[k]*(z[k]-xhatminus[k])
    P[k] = (1-KG[k])*Pminus[k]



plt.figure()
plt.plot(z,'b--',label='merenja(noisy measurment)')
plt.plot(xhat,'r-',label='procene (estimate values)')
plt.axhline(x,color='g',label='prava vrednost(true value)')
plt.legend()
plt.title('Jednostavan primer Kalmanovog filtera', fontweight='bold')
plt.xlabel('Iteracije')
plt.ylabel('Izlazni Napon')
